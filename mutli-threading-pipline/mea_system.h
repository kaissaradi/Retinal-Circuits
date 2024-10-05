#ifndef MEA_SYSTEM_H
#define MEA_SYSTEM_H

#include <stdint.h>
#include <stdbool.h>

// Configuration structure (can be loaded from a file)
typedef struct {
    uint16_t num_channels;
    uint32_t sampling_rate;
    // Add more configuration parameters as needed
} MEAConfig;

// Raw MEA data structure
typedef struct {
    uint16_t num_channels;
    uint32_t num_samples;
    float* data;
    uint64_t timestamp;
} MEARawData;

// Processed MEA data structure
typedef struct {
    uint32_t num_samples;
    float* data;
    uint64_t timestamp;
    // Add more fields for processed data as needed
} MEAProcessedData;

// Function prototypes for each module
bool mea_init(const MEAConfig* config);
void mea_cleanup(void);

// Data Acquisition Module
bool mea_acquisition_start(void);
void mea_acquisition_stop(void);
MEARawData* mea_acquisition_get_data(void);

// Data Processing Module
bool mea_processing_start(void);
void mea_processing_stop(void);
MEAProcessedData* mea_processing_get_data(void);

// Data Storage Module
bool mea_storage_start(void);
void mea_storage_stop(void);
bool mea_storage_save_data(const MEAProcessedData* data);

// Visualization Module
bool mea_visualization_start(void);
void mea_visualization_stop(void);
void mea_visualization_update(const MEAProcessedData* data);

#endif // MEA_SYSTEM_H