/**
 * @file    Utils.c
 * @brief   Utility function implementations
 *
 * @author  FedVibroSense Project
 * @version 1.0.0
 */

#include "Utils.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "stm32f4xx_hal.h"

/* =========================================================================
 * GLOBAL LOG BUFFER (declared extern in Utils.h)
 * ========================================================================= */

char _utils_log_buf[DEBUG_LOG_BUF_SIZE];

/* =========================================================================
 * PRNG STATE — LCG (Numerical Recipes parameters, full-period)
 * ========================================================================= */

static uint32_t _lcg_state = 0x12345678UL;

/* =========================================================================
 * UART LOG WRITE
 * ========================================================================= */

/**
 * @brief  Transmit a formatted log string over the debug UART.
 *
 * Uses HAL_UART_Transmit in blocking mode.  Timeout is short
 * (UART_TX_TIMEOUT_MS) — if the host is not listening, the call
 * returns quickly and the log line is silently dropped rather than
 * stalling the real-time loop.
 *
 * @param  buf  Null-terminated (or length-bounded) string
 * @param  len  Number of bytes to transmit
 */
void Utils_LogWrite(const char *buf, uint16_t len)
{
    /* huart2 is the debug UART (PA2=TX, PA3=RX), initialized by CubeMX */
    HAL_UART_Transmit(&huart2, (uint8_t *)buf, len, UART_TX_TIMEOUT_MS);
}

/* =========================================================================
 * TIMER INITIALIZATION
 * ========================================================================= */

/**
 * @brief  Configure TIM2 as a 32-bit free-running microsecond counter.
 *
 * TIM2 is a 32-bit timer on STM32F405.  With APB1 clock = 84 MHz and
 * prescaler = 83, the counter increments at 1 MHz → 1 µs resolution.
 *
 * CubeMX should generate HAL_TIM_Base_Init for TIM2.
 * This function calls HAL_TIM_Base_Start to begin counting.
 */
void Utils_TimerInit(void)
{
    /* TIM2 is configured by CubeMX:
     *   Prescaler     = 83   (APB1 timer clock 84 MHz / 84 = 1 MHz)
     *   Counter mode  = Up
     *   Period        = 0xFFFFFFFF (32-bit max)
     *   Clock divison = No division
     * Start it here: */
    HAL_TIM_Base_Start(&htim2);
    LOG_INF("TIM2 microsecond timer started");
}

/* =========================================================================
 * VECTOR OPERATIONS
 * ========================================================================= */

/**
 * @brief  Compute L2 norm of a float vector using the Cortex-M4 FPU.
 *
 * The FPU executes FMAC (fused multiply-accumulate) in a single cycle,
 * so this loop is hardware-efficient without needing CMSIS-DSP.
 *
 * @param  v  Float array
 * @param  n  Length
 * @return L2 norm
 */
float Utils_VecNorm(const float *v, uint32_t n)
{
    float sum = 0.0f;
    for (uint32_t i = 0U; i < n; i++) {
        sum += v[i] * v[i];
    }
    return sqrtf(sum);
}

/**
 * @brief  Clip all elements of a float vector to [-clip, +clip].
 *
 * Used for gradient clipping during backpropagation to prevent
 * exploding gradients which would cause NaN propagation.
 *
 * @param  v    Float array (modified in-place)
 * @param  n    Length
 * @param  clip Clip magnitude (positive)
 */
void Utils_VecClip(float *v, uint32_t n, float clip)
{
    for (uint32_t i = 0U; i < n; i++) {
        if      (v[i] >  clip) v[i] =  clip;
        else if (v[i] < -clip) v[i] = -clip;
    }
}

/* =========================================================================
 * CRC-16/CCITT
 *
 * Polynomial: 0x1021 (x^16 + x^12 + x^5 + 1)
 * Initial value: 0xFFFF
 * No input/output reflection (non-reversed)
 * XOR out: 0x0000
 *
 * This is the standard used in XMODEM, ITU-T V.41, etc.
 * ========================================================================= */

uint16_t Utils_CRC16(const uint8_t *data, uint32_t len)
{
    uint16_t crc = FL_CRC16_INIT;
    for (uint32_t i = 0U; i < len; i++) {
        crc ^= (uint16_t)((uint16_t)data[i] << 8U);
        for (uint8_t bit = 0U; bit < 8U; bit++) {
            if (crc & 0x8000U) {
                crc = (uint16_t)((crc << 1U) ^ FL_CRC16_POLY);
            } else {
                crc <<= 1U;
            }
        }
    }
    return crc;
}

/* =========================================================================
 * MEMORY DIAGNOSTICS
 * ========================================================================= */

/* Linker-script exported symbols */
extern uint32_t _estack;   /* Top of stack (from STM32F405 linker script) */
extern uint32_t _end;      /* End of BSS / start of heap */

/**
 * @brief  Estimate free stack space by measuring the gap between
 *         the current stack pointer and the top-of-heap (_end).
 *
 * This is an approximation — stack grows downward, heap grows upward.
 * The reported "free" is the untouched region between them.
 *
 * For a more accurate watermark, fill the stack with a sentinel at boot
 * and scan for the first non-sentinel on inspection.
 */
void Utils_PrintMemoryStats(void)
{
    /* Get current stack pointer via ARM intrinsic */
    uint32_t sp = __get_MSP();

    /* _end is the first address after BSS — approximates heap start */
    uint32_t heap_start = (uint32_t)&_end;

    /* Stack top is _estack (defined in linker script) */
    uint32_t stack_top  = (uint32_t)&_estack;

    uint32_t used_stack = stack_top - sp;
    uint32_t free_gap   = sp - heap_start;

    LOG_INF("--- Memory Stats ---");
    LOG_INF("  Stack top:   0x%08lX", stack_top);
    LOG_INF("  Current SP:  0x%08lX", sp);
    LOG_INF("  Heap start:  0x%08lX", heap_start);
    LOG_INF("  Stack used:  %lu bytes", used_stack);
    LOG_INF("  Free gap:    %lu bytes", free_gap);
    LOG_INF("  NN W1 size:  %u bytes", (unsigned)(NN_INPUT_SIZE * NN_HIDDEN_SIZE * 4));
    LOG_INF("--------------------");
}

/* =========================================================================
 * PSEUDO-RANDOM NUMBER GENERATOR
 *
 * LCG parameters: multiplier = 1664525, increment = 1013904223
 * (Numerical Recipes, 32-bit full-period LCG)
 *
 * Only used for Xavier weight initialization — not cryptographically secure.
 * ========================================================================= */

void Utils_SeedRNG(uint32_t seed)
{
    _lcg_state = (seed != 0U) ? seed : 0xDEADBEEFUL;
}

/**
 * @brief  Return next pseudo-random float in [-1.0, +1.0].
 *
 * The LCG produces a 32-bit integer; we map [0, 2^32) → [-1, +1]
 * by: val = (state / 2^31) - 1.0
 */
float Utils_RandF(void)
{
    _lcg_state = (_lcg_state * 1664525UL) + 1013904223UL;
    /* Map to [0, 2) then shift to [-1, 1) */
    return ((float)_lcg_state / 2147483648.0f) - 1.0f;
}
