/**
 * @file    Config.h
 * @brief   Central configuration for FedVibroSense STM32F405 FL Client
 *
 * All compile-time constants, tunable parameters, and resource limits are
 * defined here. Changing a value in this file propagates through the entire
 * firmware without touching other source files.
 *
 * @author  FedVibroSense Project
 * @version 1.0.0
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <stdint.h>

/* =========================================================================
 * SYSTEM CLOCK & TIMING
 * ========================================================================= */

/** CPU frequency in Hz (STM32F405 max: 168 MHz) */
#define SYS_CORE_CLOCK_HZ          168000000UL

/** IMU sampling rate in Hz — must be integer divisor of SYS_CORE_CLOCK_HZ */
#define IMU_SAMPLE_RATE_HZ         100U

/** Sample period in seconds (float) */
#define IMU_SAMPLE_PERIOD_S        (1.0f / IMU_SAMPLE_RATE_HZ)

/** Feature window duration in seconds */
#define FEATURE_WINDOW_SECONDS     1U

/** Number of IMU samples per feature window */
#define FEATURE_WINDOW_SAMPLES     (IMU_SAMPLE_RATE_HZ * FEATURE_WINDOW_SECONDS)  /* 100 */

/* =========================================================================
 * FEATURE EXTRACTION
 * ========================================================================= */

/**
 * Total size of the feature vector fed to the neural network.
 *
 * Composition (per window of 100 samples):
 *   pitch[100]           = 100
 *   roll[100]            = 100
 *   gyro_mag[100]        = 100
 *   accel_mag[100]       = 100
 *   delta_gyro_mag[100]  = 100
 *   ──────────────────────────
 *   TOTAL                = 500
 */
#define FEATURE_VECTOR_SIZE        500U

/** Number of raw IMU axes stored in ring buffer: ax,ay,az,gx,gy,gz */
#define IMU_RAW_AXES               6U

/** Z-score normalization epsilon to avoid division by zero */
#define ZSCORE_EPSILON             1e-7f

/* =========================================================================
 * COMPLEMENTARY FILTER
 * ========================================================================= */

/** Alpha weight for gyroscope integration (0 < alpha < 1) */
#define CF_ALPHA                   0.98f

/** Gravity constant used for accelerometer angle estimation */
#define GRAVITY_MSS                9.80665f

/* =========================================================================
 * NEURAL NETWORK TOPOLOGY
 * ========================================================================= */

#define NN_INPUT_SIZE              FEATURE_VECTOR_SIZE   /* 500 */
#define NN_HIDDEN_SIZE             16U
#define NN_OUTPUT_SIZE             3U    /* Classes: normal / imbalance / looseness */

/** Learning rate for stochastic gradient descent */
#define NN_LEARNING_RATE           0.01f

/** Xavier initializer scale: sqrt(2 / fan_in) for ReLU */
#define NN_XAVIER_SCALE_HIDDEN     0.06324555f   /* sqrt(2/500) ≈ 0.06324555 */
#define NN_XAVIER_SCALE_OUTPUT     0.35355339f   /* sqrt(2/16)  ≈ 0.35355339 */

/** Softmax temperature (1.0 = standard) */
#define NN_SOFTMAX_TEMP            1.0f

/** Gradient clipping norm to prevent exploding gradients */
#define NN_GRAD_CLIP               5.0f

/** Training iterations per FL round before weight upload */
#define FL_LOCAL_EPOCHS            10U

/* =========================================================================
 * FEDERATED LEARNING
 * ========================================================================= */

/** Total number of trainable weight parameters (for serialization) */
/* W1: 500×16 = 8000, b1: 16, W2: 16×3 = 48, b2: 3 → 8067 */
#define FL_WEIGHT_COUNT            (NN_INPUT_SIZE * NN_HIDDEN_SIZE + \
                                    NN_HIDDEN_SIZE + \
                                    NN_HIDDEN_SIZE * NN_OUTPUT_SIZE + \
                                    NN_OUTPUT_SIZE)   /* 8067 */

/** UART packet header magic bytes */
#define FL_PACKET_MAGIC_UPLOAD     0xFE01U
#define FL_PACKET_MAGIC_DOWNLOAD   0xFE02U

/** UART packet checksum polynomial (CRC-16/CCITT) */
#define FL_CRC16_POLY              0x1021U
#define FL_CRC16_INIT              0xFFFFU

/* =========================================================================
 * UART CONFIGURATION
 * ========================================================================= */

/** Debug UART baud rate */
#define UART_DEBUG_BAUD            115200U

/** FL communication UART baud rate (higher for weight transfer) */
#define UART_FL_BAUD               921600U

/** UART transmit timeout in milliseconds */
#define UART_TX_TIMEOUT_MS         100U

/** UART receive timeout in milliseconds */
#define UART_RX_TIMEOUT_MS         5000U

/** Max single UART packet payload in bytes */
#define UART_MAX_PAYLOAD_BYTES     (FL_WEIGHT_COUNT * 4U + 16U)  /* weights + header */

/* =========================================================================
 * I2C CONFIGURATION
 * ========================================================================= */

/** MPU-6050 I2C address (AD0 pin = GND → 0x68) */
#define MPU6050_I2C_ADDR           0x68U
#define MPU6050_I2C_ADDR_SHIFTED   (MPU6050_I2C_ADDR << 1)

/** I2C transaction timeout in milliseconds */
#define I2C_TIMEOUT_MS             10U

/* =========================================================================
 * MPU6050 SENSOR CONFIGURATION
 * ========================================================================= */

/** Accelerometer full-scale range: 0=±2g, 1=±4g, 2=±8g, 3=±16g */
#define MPU6050_ACCEL_FS           0U    /* ±2g → sensitivity = 16384 LSB/g */
#define MPU6050_ACCEL_SENSITIVITY  16384.0f

/** Gyroscope full-scale range: 0=±250°/s, 1=±500°/s, 2=±1000°/s, 3=±2000°/s */
#define MPU6050_GYRO_FS            0U    /* ±250°/s → sensitivity = 131 LSB/°/s */
#define MPU6050_GYRO_SENSITIVITY   131.0f

/** Number of samples averaged during calibration */
#define MPU6050_CALIBRATION_SAMPLES 200U

/** DLPF configuration: 3 = 44Hz acc / 42Hz gyro bandwidth */
#define MPU6050_DLPF_CFG           3U

/* =========================================================================
 * MEMORY & BUFFER SIZING
 * ========================================================================= */

/** Ring buffer capacity for raw IMU data (must be power-of-2 for efficiency) */
#define IMU_RING_BUFFER_DEPTH      128U   /* samples, > FEATURE_WINDOW_SAMPLES */

/** Debug log string maximum length */
#define DEBUG_LOG_BUF_SIZE         128U

/** Training loss history length (for moving average) */
#define LOSS_HISTORY_LEN           16U

/* =========================================================================
 * CLASS LABELS
 * ========================================================================= */

#define CLASS_NORMAL               0U
#define CLASS_IMBALANCE            1U
#define CLASS_LOOSENESS            2U

/* =========================================================================
 * DEBUG VERBOSITY (0=off, 1=errors, 2=info, 3=verbose)
 * ========================================================================= */

#define DEBUG_LEVEL                2U

/* =========================================================================
 * COMPILE-TIME ASSERTIONS
 * ========================================================================= */

_Static_assert(FEATURE_VECTOR_SIZE == 500U,
    "FEATURE_VECTOR_SIZE must be 500");
_Static_assert(FL_LOCAL_EPOCHS > 0U,
    "FL_LOCAL_EPOCHS must be at least 1");
_Static_assert(IMU_RING_BUFFER_DEPTH >= FEATURE_WINDOW_SAMPLES,
    "Ring buffer must hold at least one full window");

#endif /* CONFIG_H */
