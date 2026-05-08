/**
 * @file    FederatedClient.h
 * @brief   Federated Learning client state machine for STM32F405
 *
 * Implements the client-side Federated Averaging (FedAvg) protocol:
 *
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │  IDLE → COLLECTING → TRAINING → UPLOADING → DOWNLOADING → IDLE  │
 *   └─────────────────────────────────────────────────────────────────┘
 *
 *  IDLE:
 *    Waiting for enough labeled samples to fill one FL training round.
 *
 *  COLLECTING:
 *    Ring buffer accumulates feature windows.  When FL_LOCAL_EPOCHS
 *    windows have been labeled, transitions to TRAINING.
 *    (Labels are injected via FLC_SubmitLabel() — e.g., from a button
 *     press or UART command during field deployment.)
 *
 *  TRAINING:
 *    Runs NN_TrainStep() for each collected (feature, label) pair.
 *    Logs loss per epoch via UART.
 *
 *  UPLOADING:
 *    Serializes weights and transmits the FL packet over UART.
 *    Waits for ACK from aggregation server.
 *
 *  DOWNLOADING:
 *    Receives updated global model from aggregation server.
 *    Deserializes and applies weights to the local NN.
 *
 * Communication protocol (UART):
 *    Upload:   STM32 → Server  [binary FL packet, SER_TOTAL_PACKET_BYTES]
 *    ACK:      Server → STM32  [2 bytes: 0xAC 0xAC]
 *    Download: Server → STM32  [binary FL packet, SER_TOTAL_PACKET_BYTES]
 *    NACK:     Server → STM32  [2 bytes: 0xNA 0xCK = 0xNA 0xCE] (error)
 *
 * @author  FedVibroSense Project
 * @version 1.0.0
 */

#ifndef FEDERATEDCLIENT_H
#define FEDERATEDCLIENT_H

#include <stdint.h>
#include "Config.h"
#include "NeuralNetwork.h"
#include "Serialization.h"
#include "FeatureExtractor.h"
#include "stm32f4xx_hal.h"

/* =========================================================================
 * FL CLIENT FSM STATES
 * ========================================================================= */

typedef enum {
    FLC_STATE_IDLE        = 0,
    FLC_STATE_COLLECTING  = 1,
    FLC_STATE_TRAINING    = 2,
    FLC_STATE_UPLOADING   = 3,
    FLC_STATE_DOWNLOADING = 4,
    FLC_STATE_ERROR       = 5,
} FLC_State_t;

/* =========================================================================
 * ACK / NACK PROTOCOL BYTES
 * ========================================================================= */

#define FLC_ACK_BYTE0   0xACU
#define FLC_ACK_BYTE1   0xACU
#define FLC_NACK_BYTE0  0xNAU    /* 0xNA = 0x0A in practice */
#define FLC_NACK_BYTE1  0xCEU

/* =========================================================================
 * TRAINING BUFFER
 *
 * Stores FL_LOCAL_EPOCHS feature vectors and their labels.
 * Memory: FL_LOCAL_EPOCHS × (FEATURE_VECTOR_SIZE × 4 + 1)
 *       = 10 × (2000 + 1) = 20010 bytes
 * ========================================================================= */

typedef struct {
    float   features[FL_LOCAL_EPOCHS][FEATURE_VECTOR_SIZE]; /**< Stored windows */
    uint8_t labels[FL_LOCAL_EPOCHS];                        /**< Corresponding labels */
    uint8_t count;                                          /**< Samples stored so far */
} FLC_TrainingBuffer_t;

/* =========================================================================
 * FL CLIENT HANDLE
 * ========================================================================= */

typedef struct {
    FLC_State_t           state;            /**< Current FSM state */
    NN_Handle_t           *nn;              /**< Pointer to shared NN handle */
    UART_HandleTypeDef    *huart_fl;        /**< UART for FL comms (huart1) */
    FLC_TrainingBuffer_t  train_buf;        /**< Local training data buffer */
    uint32_t              round_count;      /**< FL rounds completed */
    uint32_t              total_samples;    /**< Total samples trained on */
    float                 last_round_loss;  /**< Mean loss of last FL round */
    uint8_t               server_connected; /**< 1 if server responded last round */
    uint8_t               error_code;       /**< SER_Status_t or UART error */
} FLC_Handle_t;

/* =========================================================================
 * PUBLIC API
 * ========================================================================= */

/**
 * @brief  Initialize the FL client handle.
 * @param  flc     Pointer to caller-allocated FLC_Handle_t
 * @param  nn      Shared neural network handle (must be NN_Init'd first)
 * @param  huart   UART handle for FL communication
 */
void FLC_Init(FLC_Handle_t *flc, NN_Handle_t *nn, UART_HandleTypeDef *huart);

/**
 * @brief  Submit a labeled feature window to the FL client.
 *
 * Call this after FE_BuildFeatureVector() produces a new window AND
 * a label is known (e.g., from operator annotation or supervised mode).
 *
 * When train_buf.count reaches FL_LOCAL_EPOCHS, the FSM advances to
 * FLC_STATE_TRAINING automatically.
 *
 * @param  flc      FL client handle
 * @param  feature  Float array of FEATURE_VECTOR_SIZE
 * @param  label    Ground-truth class (CLASS_NORMAL/IMBALANCE/LOOSENESS)
 * @return 1 if sample accepted, 0 if buffer full or wrong state
 */
uint8_t FLC_SubmitSample(FLC_Handle_t *flc,
                          const float feature[FEATURE_VECTOR_SIZE],
                          uint8_t label);

/**
 * @brief  Run one FSM tick — call from the main loop.
 *
 * This is the primary driver function. It advances the FSM one step:
 *   - In COLLECTING: no-op (waits for FLC_SubmitSample calls)
 *   - In TRAINING:   runs all local epoch training steps
 *   - In UPLOADING:  serializes and transmits weights, waits for ACK
 *   - In DOWNLOADING: receives and applies global model
 *   - In ERROR:      logs the error, resets to IDLE
 *
 * Non-blocking in COLLECTING; blocking during TRAINING and comms phases.
 *
 * @param  flc  FL client handle
 */
void FLC_Tick(FLC_Handle_t *flc);

/**
 * @brief  Force the FL client to attempt weight upload immediately.
 *         Useful for triggered synchronization (e.g., via UART command).
 * @param  flc  FL client handle
 */
void FLC_TriggerUpload(FLC_Handle_t *flc);

/**
 * @brief  Reset the FL client to IDLE and clear training buffer.
 * @param  flc  FL client handle
 */
void FLC_Reset(FLC_Handle_t *flc);

/**
 * @brief  Return current FSM state string for debug logging.
 * @param  state  FLC_State_t value
 * @return const char* human-readable state name
 */
const char *FLC_StateStr(FLC_State_t state);

/**
 * @brief  Print FL client status summary to UART debug port.
 * @param  flc  FL client handle
 */
void FLC_PrintStatus(const FLC_Handle_t *flc);

/* =========================================================================
 * INTERNAL HELPER DECLARATIONS (used in FederatedClient.c only)
 * ========================================================================= */

/** @internal Execute local training phase */
void FLC_DoTraining(FLC_Handle_t *flc);

/** @internal Execute weight upload phase */
void FLC_DoUpload(FLC_Handle_t *flc);

/** @internal Execute global model download phase */
void FLC_DoDownload(FLC_Handle_t *flc);

#endif /* FEDERATEDCLIENT_H */
