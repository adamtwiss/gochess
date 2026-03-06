/*
 * tbconfig.h - Configuration for Fathom Syzygy tablebase probing.
 * Customized for gochess engine integration.
 */

#ifndef TBCONFIG_H
#define TBCONFIG_H

/* Use compiler builtins for popcount and lsb */
#define TB_CUSTOM_POP_COUNT(x) (__builtin_popcountll(x))
#define TB_CUSTOM_LSB(x) (__builtin_ctzll((x)))

/* Disable helper API - we have our own attack generation */
#define TB_NO_HELPER_API

/* Scoring constants matching our engine */
#define TB_VALUE_PAWN 100
#define TB_VALUE_MATE 32000
#define TB_VALUE_INFINITE 32767
#define TB_VALUE_DRAW 0
#define TB_MAX_MATE_PLY 255

#endif
