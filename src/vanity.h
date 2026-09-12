// Host-only helpers shared by the CPU searcher, CUDA host code, and tests.
#pragma once
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include "pipeline.h"

static const char VK_B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// An ed25519 SSH pubkey blob is 19 header bytes + 32 key bytes = 51 bytes
// = 68 base64 chars, no padding. A suffix of L chars pins the low 6L bits
// of the 32-byte public key read as a big-endian integer.
// Returns the index of the first constrained byte, or -1 on bad input.
static int vk_suffix_to_target(const char *suffix, uint8_t target[32], uint8_t mask[32]) {
    size_t len = strlen(suffix);
    if (len == 0 || len > 42) return -1;
    memset(target, 0, 32);
    memset(mask, 0, 32);
    // accumulate the 6L-bit value into the tail of the 32-byte big-endian array
    for (size_t i = 0; i < len; i++) {
        const char *p = strchr(VK_B64, suffix[i]);
        if (!p || !suffix[i]) return -1;
        int val = (int)(p - VK_B64);
        // target = target << 6 | val  (over 32 bytes, big-endian)
        int carry = val;
        for (int j = 31; j >= 0; j--) {
            int t = (target[j] << 6) | carry;
            target[j] = (uint8_t)t;
            carry = t >> 8;
        }
    }
    int bits = (int)(6 * len);
    for (int j = 31; j >= 0 && bits > 0; j--) {
        mask[j] = bits >= 8 ? 0xff : (uint8_t)((1 << bits) - 1);
        bits -= 8;
    }
    for (int j = 0; j < 32; j++)
        if (mask[j]) return j;
    return -1;
}

// --- Multi-target matcher compilation (AUDIT F2) -------------------------
//
// A hunt can accept many suffixes at once (several words, and/or every case
// variant of each letter). At startup they are compiled to:
//   - one shared mask over w3 (pubkey bytes 24..31 as an LE u64; a suffix of
//     L <= 10 chars constrains only bits inside w3),
//   - a sorted, deduped list of w3 target values (sign bit included),
//   - an 8 KB bitmap over the low 16 bits of the targets.
// The kernel tests bitmap[low16(w3y & mask)] per key (~free) and only bitmap
// hits (~2^-13 of keys) pay for the x-parity and a binary search.

#include <algorithm>

struct vk_targets {
    uint64_t mask_w3full = 0;         // includes the x-sign bit when constrained
    std::vector<uint64_t> vals;       // sorted unique w3 values, pre-masked
    uint32_t bitmap[2048] = {};       // 2^16 bits over vals' low 16 bits
    size_t suffix_len = 0;
};

static void vk_expand_ci(const std::string &suffix, std::vector<std::string> &out) {
    std::vector<std::string> acc{""};
    for (char ch : suffix) {
        std::vector<std::string> next;
        next.reserve(acc.size() * 2);
        const bool alpha = (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z');
        for (const auto &v : acc) {
            if (alpha) {
                next.push_back(v + (char)toupper(ch));
                next.push_back(v + (char)tolower(ch));
            } else {
                next.push_back(v + ch);
            }
        }
        acc.swap(next);
    }
    out.insert(out.end(), acc.begin(), acc.end());
}

// Returns 0 on success; prints the reason and returns -1 on invalid input.
static int vk_compile_targets(const std::vector<std::string> &suffixes, bool ci, vk_targets &tg) {
    if (suffixes.empty()) return -1;
    tg.suffix_len = suffixes[0].size();
    for (const auto &s : suffixes)
        if (s.size() != tg.suffix_len) {
            fprintf(stderr, "all suffixes must have the same length ('%s')\n", s.c_str());
            return -1;
        }
    if (tg.suffix_len > 10) {
        fprintf(stderr, "multi-target matching supports suffixes up to 10 chars (60 bits)\n");
        return -1;
    }
    std::vector<std::string> variants;
    for (const auto &s : suffixes) {
        if (ci) vk_expand_ci(s, variants);
        else variants.push_back(s);
        if (variants.size() > (1u << 20)) {
            fprintf(stderr, "too many target variants (max 2^20)\n");
            return -1;
        }
    }
    for (const auto &v : variants) {
        uint8_t t[32], m[32];
        if (vk_suffix_to_target(v.c_str(), t, m) < 0) {
            fprintf(stderr, "bad suffix '%s'\n", v.c_str());
            return -1;
        }
        uint64_t tw = 0, mw = 0;
        for (int j = 0; j < 8; j++) {
            tw |= (uint64_t)t[24 + j] << (8 * j);
            mw |= (uint64_t)m[24 + j] << (8 * j);
        }
        tg.mask_w3full = mw;
        tg.vals.push_back(tw & mw);
    }
    std::sort(tg.vals.begin(), tg.vals.end());
    tg.vals.erase(std::unique(tg.vals.begin(), tg.vals.end()), tg.vals.end());
    for (uint64_t v : tg.vals) {
        uint32_t lo = (uint32_t)(v & 0xffff);
        tg.bitmap[lo >> 5] |= 1u << (lo & 31);
    }
    return 0;
}

// Loads either table format; file size discriminates. *wide is set to 1 for
// the 16-bit signed comb (~48 MB), 0 for the 8-bit comb (768 KB).
static std::vector<uint8_t> vk_load_table(const char *path, int *wide) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open table %s (run: python3 tools/gen_table.py %s [--wide])\n", path, path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz == VK_TABLE_BYTES) *wide = 0;
    else if (sz == VK_TABLE16_BYTES) *wide = 1;
    else {
        fprintf(stderr, "table %s: unexpected size %ld (want %d or %d)\n",
                path, sz, VK_TABLE_BYTES, VK_TABLE16_BYTES);
        exit(1);
    }
    std::vector<uint8_t> table((size_t)sz);
    size_t n = fread(table.data(), 1, table.size(), f);
    fclose(f);
    if (n != table.size()) {
        fprintf(stderr, "table %s: short read\n", path);
        exit(1);
    }
    return table;
}

static std::string vk_hex(const uint8_t *b, size_t n) {
    std::string s(2 * n, '0');
    for (size_t i = 0; i < n; i++) snprintf(&s[2 * i], 3, "%02x", b[i]);
    return s;
}

static bool vk_unhex(const char *hex, uint8_t *out, size_t n) {
    if (strlen(hex) != 2 * n) return false;
    for (size_t i = 0; i < n; i++) {
        unsigned v;
        if (sscanf(hex + 2 * i, "%2x", &v) != 1) return false;
        out[i] = (uint8_t)v;
    }
    return true;
}

// authorized_keys line for a raw 32-byte public key
static std::string vk_pub_line(const uint8_t pub[32]) {
    static const uint8_t header[19] = {0, 0, 0, 11, 's', 's', 'h', '-', 'e', 'd',
                                       '2', '5', '5', '1', '9', 0, 0, 0, 32};
    uint8_t blob[51];
    memcpy(blob, header, 19);
    memcpy(blob + 19, pub, 32);
    std::string out = "ssh-ed25519 ";
    for (int i = 0; i < 51; i += 3) {
        uint32_t v = ((uint32_t)blob[i] << 16) | ((uint32_t)blob[i + 1] << 8) | blob[i + 2];
        out += VK_B64[(v >> 18) & 63];
        out += VK_B64[(v >> 12) & 63];
        out += VK_B64[(v >> 6) & 63];
        out += VK_B64[v & 63];
    }
    return out;
}

static void vk_random_base(uint8_t base[16]) {
    FILE *f = fopen("/dev/urandom", "rb");
    if (!f || fread(base, 1, 16, f) != 16) {
        fprintf(stderr, "cannot read /dev/urandom\n");
        exit(1);
    }
    fclose(f);
}
