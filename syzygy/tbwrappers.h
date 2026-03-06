/*
 * tbwrappers.h - Go-callable wrapper functions for Fathom C macros.
 * CGO cannot call C macros directly, so we wrap them in inline functions.
 */

#ifndef TBWRAPPERS_H
#define TBWRAPPERS_H

#include "tbprobe.h"

static inline unsigned tb_get_wdl(unsigned res) {
    return TB_GET_WDL(res);
}

static inline unsigned tb_get_to(unsigned res) {
    return TB_GET_TO(res);
}

static inline unsigned tb_get_from(unsigned res) {
    return TB_GET_FROM(res);
}

static inline unsigned tb_get_promotes(unsigned res) {
    return TB_GET_PROMOTES(res);
}

static inline unsigned tb_get_dtz(unsigned res) {
    return TB_GET_DTZ(res);
}

#endif
