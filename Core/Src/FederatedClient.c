/**
 * @file    FederatedClient.c
 * @brief   Federated Learning client finite-state machine
 *
 * FL Round lifecycle:
 *
 *  1. IDLE → COLLECTING
 *     Transitions when FLC_SubmitSample() is first called.
 *
 *  2. COLLECTING → TRAINING
 *     Transitions automatically when FL_LOCAL_EPOCHS labeled samples
 *     have been accumulated in train_buf.
 *
 *  3. TRAINING (in FLC_Tick):
 *     For each sample in train_buf, runs NN_TrainStep().
 *     Logs loss per step.  Transitions to UPLOADING when done.
 *
 *  4. UPLOADING (in FLC_Tick):
 *     Calls SER_SerializeWeights() → HAL_UART_Transmit().
 *     Waits up to UART_RX_TIMEOUT_MS for a 2-byte ACK.
 *     On ACK: transitions to DOWNLOADING.
 *     On NACK or timeout: transitions to ERROR.
 *
 *  5. DOWNLOADING (in FLC_Tick):
 *     Receives SER_TOTAL_PACKET_BYTES over UART.
 *     Calls SER_DeserializeWeights() to apply global model.
 *     On success: clears train_buf, transitions to IDLE (next round).
 *     On failure: transitions to ERROR.
 *
 *  6. ERROR:
 *     Logs error code, resets to IDLE after one tick.
 *
 * @author  FedVibroSense Project
 * @version 1.0.0
 */

#include "FederatedClient.h"
#include "Utils.h"
#include <string.h>

/* ACK/NACK sequence from aggregation server */
static const uint8_t _ack_seq[2]  = { FLC_ACK_BYTE0,  FLC_ACK_BYTE1  };

/* =========================================================================
 * STATE STRING HELPER
 * ========================================================================= */

const char *FLC_StateStr(FLC_State_t state)
{
    switch (state) {
        case FLC_STATE_IDLE:        return "IDLE";
        case FLC_STATE_COLLECTING:  return "COLLECTING";
        case FLC_STATE_TRAINING:    return "TRAINING";
        case FLC_STATE_UPLOADING:   return "UPLOADING";
        case FLC_STATE_DOWNLOADING: return "DOWNLOADING";
        case FLC_STATE_ERROR:       return "ERROR";
        default:                    return "UNKNOWN";
    }
}

/* =========================================================================
 * INITIALIZATION
 * ========================================================================= */

void FLC_Init(FLC_Handle_t *flc, NN_Handle_t *nn, UART_HandleTypeDef *huart)
{
    memset(flc, 0, sizeof(FLC_Handle_t));
    flc->nn         = nn;
    flc->huart_fl   = huart;
    flc->state      = FLC_STATE_IDLE;
    LOG_INF("FL client initialized (FL_LOCAL_EPOCHS=%u)", FL_LOCAL_EPOCHS);
}

/* =========================================================================
 * SAMPLE SUBMISSION
 * ========================================================================= */

/**
 * @brief  Accept a labeled feature window and buffer it for training.
 *
 * In a real deployment, labels are provided by:
 *  a) An operator pressing a button to annotate the current condition
 *  b) A UART command from a host ('0' = normal, '1' = imbalance, '2' = looseness)
 *  c) A semi-supervised scheme using pseudo-labels from inference
 *
 * This firmware accepts the label as a parameter — the labeling mechanism
 * is left to the application layer in main.c.
 */
uint8_t FLC_SubmitSample(FLC_Handle_t *flc,
                          const float feature[FEATURE_VECTOR_SIZE],
                          uint8_t label)
{
    /* Accept samples in IDLE or COLLECTING states */
    if (flc->state != FLC_STATE_IDLE &&
        flc->state != FLC_STATE_COLLECTING) {
        return 0U;
    }

    if (flc->train_buf.count >= FL_LOCAL_EPOCHS) {
        LOG_ERR("FL: train buffer full, ignoring sample");
        return 0U;
    }

    /* Transition to COLLECTING on first sample */
    if (flc->state == FLC_STATE_IDLE) {
        flc->state = FLC_STATE_COLLECTING;
        LOG_INF("FL: IDLE → COLLECTING");
    }

    /* Copy feature vector and label */
    uint8_t idx = flc->train_buf.count;
    memcpy(flc->train_buf.features[idx], feature,
           FEATURE_VECTOR_SIZE * sizeof(float));
    flc->train_buf.labels[idx] = label;
    flc->train_buf.count++;

    LOG_VRB("FL: buffered sample %u/%u, label=%u",
            flc->train_buf.count, FL_LOCAL_EPOCHS, label);

    /* When buffer is full, advance to TRAINING */
    if (flc->train_buf.count >= FL_LOCAL_EPOCHS) {
        flc->state = FLC_STATE_TRAINING;
        LOG_INF("FL: COLLECTING → TRAINING (%u samples ready)", FL_LOCAL_EPOCHS);
    }

    return 1U;
}

/* =========================================================================
 * FSM TICK — main driver, call from main loop
 * ========================================================================= */

void FLC_Tick(FLC_Handle_t *flc)
{
    switch (flc->state) {
        case FLC_STATE_IDLE:
        case FLC_STATE_COLLECTING:
            /* Waiting for samples — nothing to do */
            break;

        case FLC_STATE_TRAINING:
            FLC_DoTraining(flc);
            break;

        case FLC_STATE_UPLOADING:
            FLC_DoUpload(flc);
            break;

        case FLC_STATE_DOWNLOADING:
            FLC_DoDownload(flc);
            break;

        case FLC_STATE_ERROR:
            LOG_ERR("FL: error code %u, resetting to IDLE", flc->error_code);
            FLC_Reset(flc);
            break;

        default:
            LOG_ERR("FL: unknown state %u", (unsigned)flc->state);
            flc->state = FLC_STATE_ERROR;
            break;
    }
}

/* =========================================================================
 * TRAINING PHASE
 * ========================================================================= */

/**
 * @brief  Run all local training steps, then advance to UPLOADING.
 *
 * For each sample in train_buf:
 *   - Call NN_TrainStep() with the pre-computed feature vector and label
 *   - Log loss every 5 steps
 *
 * After all steps: compute mean loss, log it, transition to UPLOADING.
 */
void FLC_DoTraining(FLC_Handle_t *flc)
{
    LOG_INF("FL: === LOCAL TRAINING START (round %lu) ===",
            flc->round_count + 1UL);

    float total_loss = 0.0f;

    for (uint8_t s = 0U; s < flc->train_buf.count; s++) {
        float loss = NN_TrainStep(flc->nn,
                                  flc->train_buf.features[s],
                                  flc->train_buf.labels[s]);
        total_loss += loss;

        LOG_INF("FL: train step %u/%u label=%u loss=%.4f",
                s + 1U, flc->train_buf.count,
                flc->train_buf.labels[s], (double)loss);
    }

    flc->last_round_loss = total_loss / (float)flc->train_buf.count;
    flc->total_samples  += flc->train_buf.count;

    LOG_INF("FL: training done, mean_loss=%.4f, total_samples=%lu",
            (double)flc->last_round_loss, flc->total_samples);
    LOG_INF("FL: TRAINING → UPLOADING");

    flc->state = FLC_STATE_UPLOADING;
}

/* =========================================================================
 * UPLOAD PHASE
 * ========================================================================= */

/**
 * @brief  Serialize local model and transmit over UART to aggregation server.
 *
 * Protocol:
 *  STM32 → Server:  FL packet (SER_TOTAL_PACKET_BYTES bytes)
 *  Server → STM32:  ACK (2 bytes: 0xAC 0xAC) or NACK on error
 *
 * On ACK:  advance to DOWNLOADING
 * On NACK: advance to ERROR
 * On UART timeout: advance to ERROR
 *
 * Uses the shared ser_packet_buf (declared in Serialization.c).
 * Total transfer: ~32 KB at 921600 baud ≈ 352 ms.
 */
void FLC_DoUpload(FLC_Handle_t *flc)
{
    LOG_INF("FL: UPLOADING weights (%u bytes)...", SER_TOTAL_PACKET_BYTES);

    /* Step 1: serialize weights into packet buffer */
    SER_Status_t ser_status = SER_SerializeWeights(
        flc->nn,
        FL_PACKET_MAGIC_UPLOAD,
        ser_packet_buf,
        SER_TOTAL_PACKET_BYTES
    );

    if (ser_status != SER_OK) {
        LOG_ERR("FL: serialization failed, code=%u", (unsigned)ser_status);
        flc->error_code = (uint8_t)ser_status;
        flc->state = FLC_STATE_ERROR;
        return;
    }

    /* Step 2: transmit packet over FL UART */
    HAL_StatusTypeDef tx_ret = HAL_UART_Transmit(
        flc->huart_fl,
        ser_packet_buf,
        SER_TOTAL_PACKET_BYTES,
        UART_TX_TIMEOUT_MS * 100U   /* Allow up to 10 s for large packet */
    );

    if (tx_ret != HAL_OK) {
        LOG_ERR("FL: UART transmit failed, HAL=%d", (int)tx_ret);
        flc->error_code = 0xFFU;
        flc->state = FLC_STATE_ERROR;
        return;
    }

    LOG_INF("FL: upload TX complete, waiting for ACK...");

    /* Step 3: wait for 2-byte ACK from server */
    uint8_t ack_buf[2] = {0U, 0U};
    HAL_StatusTypeDef rx_ret = HAL_UART_Receive(
        flc->huart_fl,
        ack_buf,
        2U,
        UART_RX_TIMEOUT_MS
    );

    if (rx_ret != HAL_OK) {
        LOG_ERR("FL: ACK timeout or UART error (HAL=%d)", (int)rx_ret);
        flc->server_connected = 0U;
        flc->error_code = 0xFEU;
        flc->state = FLC_STATE_ERROR;
        return;
    }

    /* Check ACK sequence */
    if (ack_buf[0] != _ack_seq[0] || ack_buf[1] != _ack_seq[1]) {
        LOG_ERR("FL: bad ACK [0x%02X 0x%02X]", ack_buf[0], ack_buf[1]);
        flc->server_connected = 0U;
        flc->error_code = 0xFDU;
        flc->state = FLC_STATE_ERROR;
        return;
    }

    flc->server_connected = 1U;
    LOG_INF("FL: ACK received — UPLOADING → DOWNLOADING");
    flc->state = FLC_STATE_DOWNLOADING;
}

/* =========================================================================
 * DOWNLOAD PHASE
 * ========================================================================= */

/**
 * @brief  Receive aggregated global model and apply to local NN.
 *
 * Protocol:
 *  Server → STM32: FL packet (SER_TOTAL_PACKET_BYTES bytes)
 *
 * On success: apply weights, clear train_buf, increment round, → IDLE
 * On CRC/format error: log error, revert to local weights, → ERROR
 *
 * The receive timeout is generous (UART_RX_TIMEOUT_MS * 100) to account
 * for the server needing time to perform FedAvg aggregation.
 */
void FLC_DoDownload(FLC_Handle_t *flc)
{
    LOG_INF("FL: DOWNLOADING global model (%u bytes)...",
            SER_TOTAL_PACKET_BYTES);

    /* Receive the global model packet */
    HAL_StatusTypeDef rx_ret = HAL_UART_Receive(
        flc->huart_fl,
        ser_packet_buf,
        SER_TOTAL_PACKET_BYTES,
        UART_RX_TIMEOUT_MS * 100U
    );

    if (rx_ret != HAL_OK) {
        LOG_ERR("FL: download RX timeout or error (HAL=%d)", (int)rx_ret);
        flc->error_code = 0xFCU;
        flc->state = FLC_STATE_ERROR;
        return;
    }

    /* Deserialize and apply to NN */
    SER_Status_t deser_status = SER_DeserializeWeights(
        flc->nn,
        ser_packet_buf,
        SER_TOTAL_PACKET_BYTES
    );

    if (deser_status != SER_OK) {
        LOG_ERR("FL: deserialization failed, code=%u", (unsigned)deser_status);
        flc->error_code = (uint8_t)deser_status;
        flc->state = FLC_STATE_ERROR;
        return;
    }

    /* Round complete — reset training buffer and advance round counter */
    flc->round_count++;
    memset(&flc->train_buf, 0, sizeof(FLC_TrainingBuffer_t));

    LOG_INF("FL: === ROUND %lu COMPLETE ===", flc->round_count);
    LOG_INF("FL: DOWNLOADING → IDLE");
    flc->state = FLC_STATE_IDLE;

    FLC_PrintStatus(flc);
}

/* =========================================================================
 * UTILITY
 * ========================================================================= */

void FLC_TriggerUpload(FLC_Handle_t *flc)
{
    if (flc->state == FLC_STATE_COLLECTING ||
        flc->state == FLC_STATE_IDLE) {
        LOG_INF("FL: manual upload triggered");
        flc->state = FLC_STATE_UPLOADING;
    }
}

void FLC_Reset(FLC_Handle_t *flc)
{
    memset(&flc->train_buf, 0, sizeof(FLC_TrainingBuffer_t));
    flc->state      = FLC_STATE_IDLE;
    flc->error_code = 0U;
    LOG_INF("FL: reset to IDLE");
}

void FLC_PrintStatus(const FLC_Handle_t *flc)
{
    LOG_INF("FL Status: state=%s round=%lu samples=%lu loss=%.4f server=%u",
            FLC_StateStr(flc->state),
            flc->round_count,
            flc->total_samples,
            (double)flc->last_round_loss,
            flc->server_connected);
    NN_PrintWeightStats(flc->nn);
}
