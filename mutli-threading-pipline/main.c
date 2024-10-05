#include "mea_system.h"
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>

static volatile bool keep_running = true;

static void signal_handler(int signum) {
    keep_running = false;
}

int main(void) {
    signal(SIGINT, signal_handler);

    MEAConfig config = {
        .num_channels = 64,
        .sampling_rate = 20000
        // Set other configuration parameters
    };

    if (!mea_init(&config)) {
        fprintf(stderr, "Failed to initialize MEA system\n");
        return EXIT_FAILURE;
    }

    if (!mea_acquisition_start() || !mea_processing_start() ||
        !mea_storage_start() || !mea_visualization_start()) {
        fprintf(stderr, "Failed to start MEA system modules\n");
        mea_cleanup();
        return EXIT_FAILURE;
    }

    printf("MEA system running. Press Ctrl+C to stop.\n");

    while (keep_running) {
        MEAProcessedData* data = mea_processing_get_data();
        if (data) {
            mea_storage_save_data(data);
            mea_visualization_update(data);
        }
        // Add a small delay to prevent busy-waiting
    }

    mea_visualization_stop();
    mea_storage_stop();
    mea_processing_stop();
    mea_acquisition_stop();
    mea_cleanup();

    printf("MEA system stopped.\n");
    return EXIT_SUCCESS;
}