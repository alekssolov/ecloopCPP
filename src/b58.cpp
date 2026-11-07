/*
 * Copyright 2012-2014 Luke Dashjr
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Переписано в стиле C++ (C++17).
 */

#include "b58.hpp"
#include "sha256.hpp"   // sha256_final, для проверки префикса

#include <array>
#include <vector>
#include <string>
#include <string_view>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <algorithm>

/* Little/big-endian byte conversions */
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
# define le16(x) ((u16)(x))
# define le32(x) ((u32)(x))
# define le64(x) ((u64)(x))
# define be16(x) __builtin_bswap16(x)
# define be32(x) __builtin_bswap32(x)
# define be64(x) __builtin_bswap64(x)
#else
# define le16(x) __builtin_bswap16(x)
# define le32(x) __builtin_bswap32(x)
# define le64(x) __builtin_bswap64(x)
# define be16(x) ((u16)(x))
# define be32(x) ((u32)(x))
# define be64(x) ((u64)(x))
#endif
namespace base58 {

// ----------------------------------------------------------------------------
// Базовые типы
// ----------------------------------------------------------------------------
using Pattern20 = std::array<std::uint8_t, 20>;

// ----------------------------------------------------------------------------
// Таблицы base58
// ----------------------------------------------------------------------------
namespace tables {

inline constexpr std::array<std::int8_t, 128> b58digits_map = {
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1, 0, 1, 2, 3, 4, 5, 6,  7, 8,-1,-1,-1,-1,-1,-1,
    -1, 9,10,11,12,13,14,15, 16,-1,17,18,19,20,21,-1,
    22,23,24,25,26,27,28,29, 30,31,32,-1,-1,-1,-1,-1,
    -1,33,34,35,36,37,38,39, 40,41,42,43,-1,44,45,46,
    47,48,49,50,51,52,53,54, 55,56,57,-1,-1,-1,-1,-1,
};

inline constexpr std::string_view b58digits_ordered =
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

} // namespace tables

// ----------------------------------------------------------------------------
// Низкоуровневые функции кодирования/декодирования base58
// ----------------------------------------------------------------------------

// Декодирование base58 -> бинарные данные.
// Возвращает true при успехе; *binszp обновляется на реальное число байт.
bool decode(void* bin, std::size_t* binszp, std::string_view b58) {
    const std::size_t binsz = *binszp;
    const auto* b58u = reinterpret_cast<const unsigned char*>(b58.data());
    auto* binu = static_cast<unsigned char*>(bin);

    const std::size_t b58sz = b58.size();
    const std::size_t outisz = (binsz + 3) / 4;

    std::vector<std::uint32_t> outi(outisz, 0);

    const std::uint8_t bytesleft = static_cast<std::uint8_t>(binsz % 4);
    const std::uint32_t zeromask =
        bytesleft ? (0xffffffffu << (bytesleft * 8)) : 0u;

    unsigned zerocount = 0;
    std::size_t i = 0;

    // Ведущие нули просто считаем
    for (; i < b58sz && b58u[i] == '1'; ++i)
        ++zerocount;

    for (; i < b58sz; ++i) {
        if (b58u[i] & 0x80)
            return false; // высокий бит установлен -> недопустимый символ
        if (tables::b58digits_map[b58u[i]] == -1)
            return false; // недопустимая цифра base58

        std::uint32_t c = static_cast<std::uint32_t>(tables::b58digits_map[b58u[i]]);
        for (std::size_t j = outisz; j-- > 0; ) {
            const std::uint64_t t =
                static_cast<std::uint64_t>(outi[j]) * 58 + c;
            c = static_cast<std::uint32_t>((t & 0x3f00000000ULL) >> 32);
            outi[j] = static_cast<std::uint32_t>(t & 0xffffffffULL);
        }
        if (c)
            return false; // число слишком велико (перенос за пределы)
        if (outi[0] & zeromask)
            return false; // число слишком велико (старший int32 переполнен)
    }

    std::size_t j = 0;
    switch (bytesleft) {
        case 3:
            *(binu++) = static_cast<unsigned char>((outi[0] & 0xff0000) >> 16);
            [[fallthrough]];
        case 2:
            *(binu++) = static_cast<unsigned char>((outi[0] & 0x00ff00) >> 8);
            [[fallthrough]];
        case 1:
            *(binu++) = static_cast<unsigned char>(outi[0] & 0x0000ff);
            ++j;
            [[fallthrough]];
        default:
            break;
    }

    for (; j < outisz; ++j) {
        *(binu++) = static_cast<unsigned char>((outi[j] >> 0x18) & 0xff);
        *(binu++) = static_cast<unsigned char>((outi[j] >> 0x10) & 0xff);
        *(binu++) = static_cast<unsigned char>((outi[j] >> 0x08) & 0xff);
        *(binu++) = static_cast<unsigned char>((outi[j] >> 0x00) & 0xff);
    }

    // Подсчет канонического числа байт
    binu = static_cast<unsigned char*>(bin);
    for (i = 0; i < binsz; ++i) {
        if (binu[i])
            break;
        --*binszp;
    }
    *binszp += zerocount;

    return true;
}

// Кодирование бинарных данных -> base58.
bool encode(char* b58, const void* data, std::size_t binsz) {
    const auto* bin = static_cast<const std::uint8_t*>(data);
    std::ptrdiff_t zcount = 0;

    while (zcount < static_cast<std::ptrdiff_t>(binsz) && !bin[zcount])
        ++zcount;

    const std::size_t size =
        static_cast<std::size_t>(binsz - zcount) * 138 / 100 + 1;
    std::vector<std::uint8_t> buf(size, 0);

    std::ptrdiff_t high = static_cast<std::ptrdiff_t>(size) - 1;
    std::ptrdiff_t j = 0;
    for (std::ptrdiff_t i = zcount; i < static_cast<std::ptrdiff_t>(binsz);
         ++i, high = j) {
        int carry = bin[i];
        for (j = static_cast<std::ptrdiff_t>(size) - 1;
             (j > high) || carry; --j) {
            carry += 256 * buf[j];
            buf[j] = static_cast<std::uint8_t>(carry % 58);
            carry /= 58;
        }
    }

    for (j = 0; j < static_cast<std::ptrdiff_t>(size) && !buf[j]; ++j)
        ;

    std::ptrdiff_t i = 0;
    if (zcount)
        std::memset(b58, '1', static_cast<std::size_t>(zcount));
    for (i = zcount; j < static_cast<std::ptrdiff_t>(size); ++i, ++j)
        b58[i] = tables::b58digits_ordered[buf[j]];
    b58[i] = '\0';

    return true;
}

// ----------------------------------------------------------------------------
// Класс набора префиксов / шаблонов
// ----------------------------------------------------------------------------
class PatternSet {
public:
    struct Range {
        alignas(8) Pattern20 low;   // нижняя граница
        alignas(8) Pattern20 high;  // верхняя граница
    };

    static constexpr std::size_t kMaxPatterns = 10000;

    PatternSet() { patterns_.reserve(128); }

    // Преобразует префикс адреса в один или несколько 20-байтовых шаблонов
    // для сопоставления на двоичном уровне.
    bool add_prefix(std::string_view prefix);

    // Оценка сложности подбора по всем накопленным шаблонам.
    [[nodiscard]] double get_difficulty() const;

    [[nodiscard]] std::size_t size() const noexcept { return patterns_.size(); }
    [[nodiscard]] const std::vector<Range>& patterns() const noexcept {
        return patterns_;
    }

private:
    void add_pattern(const std::uint8_t* low, const std::uint8_t* high);

    // То же, но отбрасывает значения, не дающие нужного префикса.
    void add_pattern_checked(std::uint8_t* low, std::uint8_t* high,
                             std::string_view prefix);

    std::vector<Range> patterns_;
};

// ----------------------------------------------------------------------------
// Вспомогательные функции уровня файла
// ----------------------------------------------------------------------------
namespace {

inline int cmp20(const void* a, const void* b) {
    return std::memcmp(a, b, 20);
}

// Декремент 20-байтового big-endian-подобного значения (как в оригинале:
// данные интерпретируются как массив из 5 little-endian uint32).
void make_low_minus_one(const std::uint8_t* low, std::uint32_t (&out)[5]) {
    std::memcpy(out, low, 20);
    if (!out[4]--)
        if (!out[3]--)
            if (!out[2]--)
                if (!out[1]--)
                    if (!out[0]--)
                        std::memset(out, 0x00, 20);
}

void make_high_plus_one(const std::uint8_t* high, std::uint32_t (&out)[5]) {
    std::memcpy(out, high, 20);
    if (!++out[4])
        if (!++out[3])
            if (!++out[2])
                if (!++out[1])
                    if (!++out[0])
                        std::memset(out, 0xff, 20);
}

} // namespace


// ----------------------------------------------------------------------------
// Проверка префикса по настоящему адресу
// ----------------------------------------------------------------------------
// Первые 28 символов адреса определяются только значением hash160, а начиная с
// 29-го -- ещё и контрольной суммой. Узкие диапазоны появляются как раз с 29
// символов, поэтому проверить их без настоящей суммы нельзя.
namespace {

// sha256_final в проекте принимает ЦЕЛЫЕ 64-байтовые блоки и дополнение не
// делает -- его строит вызывающая сторона, как это делают prepare33/prepare65.
// Короткое сообщение поэтому дополняется здесь; иначе состояние вернётся
// нетронутым вектором инициализации, и хеш будет молча неверным.
void sha256_padded(const std::uint8_t* data, std::size_t len, std::uint8_t out[32]) {
    std::uint8_t blk[128] = {};
    std::memcpy(blk, data, len);
    blk[len] = 0x80;
    const std::size_t total = (len + 9 <= 64) ? 64 : 128;
    const std::uint64_t bits = static_cast<std::uint64_t>(len) * 8;
    for (int i = 0; i < 8; ++i)
        blk[total - 1 - i] = static_cast<std::uint8_t>(bits >> (i * 8));

    std::uint32_t st[8];
    ecloop::sha256_final(st, blk, static_cast<std::uint32_t>(total));
    for (int i = 0; i < 8; ++i) {
        out[i * 4 + 0] = static_cast<std::uint8_t>(st[i] >> 24);
        out[i * 4 + 1] = static_cast<std::uint8_t>(st[i] >> 16);
        out[i * 4 + 2] = static_cast<std::uint8_t>(st[i] >> 8);
        out[i * 4 + 3] = static_cast<std::uint8_t>(st[i]);
    }
}

// Даёт ли этот hash160 адрес, начинающийся с prefix?
bool matches_prefix(const std::uint8_t* h160, std::string_view prefix) {
    std::uint8_t body[25];
    body[0] = 0x00;                       // основная сеть, pay to public key hash
    std::memcpy(body + 1, h160, 20);

    std::uint8_t d1[32], d2[32];
    sha256_padded(body, 21, d1);
    sha256_padded(d1, 32, d2);
    std::memcpy(body + 21, d2, 4);

    std::array<char, 64> addr{};
    if (!encode(addr.data(), body, 25)) return false;
    return std::strncmp(addr.data(), prefix.data(), prefix.size()) == 0;
}

} // namespace

void PatternSet::add_pattern(const std::uint8_t* low, const std::uint8_t* high) {
    // low/high указывают на изменяемые буферы вызывающей стороны (24/25-байтовые
    // массивы), поэтому работаем с ними как с указателями.
    auto* low_p = const_cast<std::uint8_t*>(low);
    auto* high_p = const_cast<std::uint8_t*>(high);

rescan:
    std::uint32_t low_minus_one[5];
    std::uint32_t high_plus_one[5];
    make_low_minus_one(low_p, low_minus_one);
    make_high_plus_one(high_p, high_plus_one);

    // Проход по существующим шаблонам
    for (std::size_t i = 0; i < patterns_.size(); ++i) {
        Range& cur = patterns_[i];

        // Игнорируем новый шаблон, если он полностью внутри существующего
        if (cmp20(low_p, cur.low.data()) >= 0 &&
            cmp20(high_p, cur.high.data()) <= 0)
            return;

        // Расширяем существующий шаблон вниз
        if (cmp20(low_p, cur.low.data()) < 0 &&
            cmp20(high_plus_one, cur.low.data()) >= 0) {
            if (cmp20(high_p, cur.high.data()) < 0)
                std::memcpy(high_p, cur.high.data(), 20);
            patterns_.erase(patterns_.begin() + static_cast<std::ptrdiff_t>(i));
            goto rescan;
        }

        // Расширяем существующий шаблон вверх
        if (cmp20(high_p, cur.high.data()) > 0 &&
            cmp20(low_minus_one, cur.high.data()) <= 0) {
            std::memcpy(low_p, cur.low.data(), 20);
            patterns_.erase(patterns_.begin() + static_cast<std::ptrdiff_t>(i));
            goto rescan;
        }
    }

    {
        Range r{};
        std::memcpy(r.low.data(), low_p, 20);
        std::memcpy(r.high.data(), high_p, 20);
        patterns_.push_back(r);
    }

    // Подбор адресов становится очень медленным при слишком большом числе шаблонов
    if (patterns_.size() > kMaxPatterns)
        throw std::runtime_error("Error: Too many patterns!");
}


// Кладёт диапазон, отбросив значения, которые префиксу не соответствуют.
//
// Границы, которые строит add_prefix, -- НАДмножество решений, взятое с
// запасом. Пока диапазон широк, запас теряется в шуме. Но с 28 символов в нём
// остаются единицы значений, и лишнее там искажает ответ в разы, а на
// недостижимом префиксе даёт конечную сложность вместо честной бесконечности.
//
// Поэтому узкий диапазон перебирается поштучно, и в набор попадают только
// подходящие значения. Широкий добавляется как есть: перебирать 10^36 значений
// незачем, да и запас там относительно ширины исчезающе мал.
void PatternSet::add_pattern_checked(std::uint8_t* low, std::uint8_t* high,
                                     std::string_view prefix) {
    // Ширина считается, только если старшие 18 байт совпали; иначе диапазон
    // заведомо шире 2^16 и поштучный перебор бессмыслен.
    if (std::memcmp(low, high, 18) != 0) {
        add_pattern(low, high);
        return;
    }
    const unsigned lo = (static_cast<unsigned>(low[18]) << 8) | low[19];
    const unsigned hi = (static_cast<unsigned>(high[18]) << 8) | high[19];
    if (hi < lo) return;                  // пустой диапазон

    for (unsigned x = lo; x <= hi; ++x) {
        // ДВА буфера, не один. add_pattern при слиянии диапазонов пишет в свои
        // аргументы через const_cast: передать один и тот же массив и как low,
        // и как high означает испортить его на первом же слиянии, после чего
        // весь остаток перебора идёт по мусору.
        std::uint8_t vlo[20], vhi[20];
        std::memcpy(vlo, low, 20);
        vlo[18] = static_cast<std::uint8_t>(x >> 8);
        vlo[19] = static_cast<std::uint8_t>(x);
        if (!matches_prefix(vlo, prefix)) continue;
        std::memcpy(vhi, vlo, 20);
        add_pattern(vlo, vhi);
    }
}

bool PatternSet::add_prefix(std::string_view prefix) {
    std::size_t pattern_sz = 25;
    std::size_t b58sz = prefix.size();

    std::array<std::uint8_t, 32> pattern1{};
    std::array<std::uint8_t, 32> pattern2{};
    std::array<char, 128> pat1{};
    std::array<char, 128> pat2{};
    std::array<char, 128> test{};

    const int plen = static_cast<int>(prefix.size());
    int offset = 0;
    int i = 0, j = 0, nonzero = 0;
    bool lt = false;

    // Проверка префикса
    if (prefix.empty() || prefix[0] != '1') {
        std::fprintf(stderr, "Error: Prefix must start with '1'.\n");
        return false;
    }

    // Префикс как C-строка для функций декодирования
    const std::string prefix_str(prefix);

    if (!decode(pattern1.data(), &pattern_sz, prefix_str)) {
        std::fprintf(stderr,
            "Error: Address '%s' contains an invalid character.\n",
            prefix_str.c_str());
        return false;
    }

    // Особый случай: префикс состоит только из '1'
    for (i = 0; i < plen && prefix[i] == '1'; ++i)
        ;
    if (i == plen) {
        std::memset(pattern1.data() + 1, 0, 24);
        pattern2[0] = 0;
        std::memset(pattern2.data() + 1, 0xff, 24);
        for (j = 1; j < 25; ++j) {
            encode(test.data(), pattern2.data(), 25);
            if (!std::strncmp(test.data(), prefix_str.c_str(),
                              static_cast<std::size_t>(plen)))
                break;
            pattern2[j] = 0;
        }
        add_pattern_checked(pattern1.data() + 1, pattern2.data() + 1, prefix);
        return true;
    }

    do {
        std::strcpy(pat1.data() + offset, prefix_str.c_str());
        std::strcpy(pat2.data() + offset, prefix_str.c_str());
        lt = (std::strcmp(pat1.data(), "1QLbz7JHiBTspS962RLKV8GndWFw") < 0);
        for (i = static_cast<int>(std::strlen(pat1.data())); i < (lt ? 34 : 33); ++i) {
            pat1[i] = '1';
            pat2[i] = 'z';
        }
        pat1[i] = '\0';
        pat2[i] = '\0';

        b58sz = static_cast<std::size_t>(i);
        pattern_sz = 25;
        decode(pattern1.data(), &pattern_sz,
               std::string_view(pat1.data(), b58sz));
        b58sz = static_cast<std::size_t>(i);
        pattern_sz = 25;
        decode(pattern2.data(), &pattern_sz,
               std::string_view(pat2.data(), b58sz));

        ++offset;
        encode(test.data(), pattern1.data(), 25);
    } while (offset < 34 &&
             std::strncmp(test.data(), prefix_str.c_str(),
                          static_cast<std::size_t>(plen)));

    // Поиск первого ненулевого байта в любом из шаблонов
    for (i = 0; i < 25 && !(pattern1[i] | pattern2[i]); ++i)
        ;
    nonzero = i;
    if (pattern1[nonzero]) {
        add_pattern_checked(pattern1.data() + 1, pattern2.data() + 1, prefix);
    } else {
        pattern2[nonzero] = 0;
        std::memset(pattern2.data() + nonzero + 1, 0xff,
                    static_cast<std::size_t>(20 - nonzero));
        add_pattern_checked(pattern1.data() + 1, pattern2.data() + 1, prefix);
        ++nonzero;
    }

    if (!lt)
        return true;

    std::strcpy(pat1.data() + offset, prefix_str.c_str());
    std::strcpy(pat2.data() + offset, prefix_str.c_str());
    for (i = static_cast<int>(std::strlen(pat1.data())); i < 34; ++i) {
        pat1[i] = '1';
        pat2[i] = 'z';
    }
    pat1[i] = '\0';
    pat2[i] = '\0';

    b58sz = static_cast<std::size_t>(i);
    pattern_sz = 25;
    decode(pattern1.data(), &pattern_sz, std::string_view(pat1.data(), b58sz));
    b58sz = static_cast<std::size_t>(i);
    pattern_sz = 25;
    decode(pattern2.data(), &pattern_sz, std::string_view(pat2.data(), b58sz));

    if (pattern1[nonzero] && pattern2[nonzero]) {
        add_pattern_checked(pattern1.data() + 1, pattern2.data() + 1, prefix);
    } else if (!pattern1[nonzero] && pattern2[nonzero]) {
        pattern1[nonzero] = 1;
        std::memset(pattern1.data() + nonzero + 1, 0,
                    static_cast<std::size_t>(20 - nonzero));
        for (i = nonzero - 1; i >= 0; --i)
            if (pattern2[i])
                break;
        if (i >= 0) {
            std::memset(pattern2.data(), 0, static_cast<std::size_t>(i + 1));
            std::memset(pattern2.data() + i + 1, 0xff,
                        static_cast<std::size_t>(21 - i - 1));
        }
        add_pattern_checked(pattern1.data() + 1, pattern2.data() + 1, prefix);
    }

    return true;
}

double PatternSet::get_difficulty() const {
    std::uint32_t total[5] = {0, 0, 0, 0, 0};
    std::uint64_t temp;

    // Цикл по каждому шаблону
    for (const Range& r : patterns_) {
        const auto* low = reinterpret_cast<const std::uint32_t*>(r.low.data());
        const auto* high = reinterpret_cast<const std::uint32_t*>(r.high.data());

        // total += high - low + 1
        //
        // Диапазон ВКЛЮЧАЕТ обе границы, то есть содержит high-low+1 значений.
        // Со старым выражением диапазон из одного адреса считался нулевым, freq
        // обнулялась и сложность уходила в бесконечность. Отсюда и брался предел
        // в 28 символов: начиная с 29-го префикс задаёт hash160 однозначно,
        // потому что 58^5 уже меньше тех 2^32 контрольной суммы, которыми
        // префикс не управляет.
        //
        // Прибавка сидит в слагаемом при j == 4 в виде 2 вместо 1: единица --
        // это дополнение до двух для ~low, вторая -- включающий счёт.
        temp = 0;
        for (int j = 4; j >= 0; --j) {
            temp += static_cast<std::uint64_t>(total[j]) + be32(high[j]) +
                    (~be32(low[j]) + (j == 4 ? 2u : 0u));
            total[j] = static_cast<std::uint32_t>(temp);
            temp >>= 32;
        }
    }

    // Складываем дроби, от младшего к старшему разряду
    double freq = total[4] / 1461501637330902918203684832716283019655932542976.0;
    freq += total[3] / 340282366920938463463374607431768211456.0;
    freq += total[2] / 79228162514264337593543950336.0;
    freq += total[1] / 18446744073709551616.0;
    freq += total[0] / 4294967296.0;

    return 1.0 / freq;
}

} // namespace base58

// ----------------------------------------------------------------------------
// Обертки в стиле C для совместимости с остальным кодом (externs.h)
// ----------------------------------------------------------------------------

namespace {
base58::PatternSet g_pattern_set;
} // namespace

bool b58tobin(void* bin, size_t* binszp, const char* b58, size_t b58sz) {
    const std::string_view sv =
        b58sz ? std::string_view(b58, b58sz) : std::string_view(b58);
    return base58::decode(bin, binszp, sv);
}

bool b58enc(char* b58, const void* data, size_t binsz) {
    return base58::encode(b58, data, binsz);
}

bool add_prefix(const char* prefix) {
    try {
        return g_pattern_set.add_prefix(prefix);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        std::exit(0);
    }
}

double get_difficulty() {
    return g_pattern_set.get_difficulty();
}

std::vector<PrefixRange> prefix_ranges() {
    std::vector<PrefixRange> out;
    out.reserve(g_pattern_set.size());
    for (const auto& r : g_pattern_set.patterns()) {
        PrefixRange pr;
        std::memcpy(pr.low, r.low.data(), 20);
        std::memcpy(pr.high, r.high.data(), 20);
        out.push_back(pr);
        
    }
     return out;
    
}
