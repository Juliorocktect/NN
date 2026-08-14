#include "ImageLoading.h"

namespace ImagePreProcessor
{
    int readInt(std::ifstream &ifs)
    {
        unsigned char bytes[4];
        ifs.read(reinterpret_cast<char *>(bytes), 4);
        return (int(bytes[0]) << 24) | (int(bytes[1]) << 16) | (int(bytes[2]) << 8) | int(bytes[3]);
    }
    double *loadImages()
    {
        std::ifstream imageFile(IMAGE_PATH, std::ios::binary);
        if (!imageFile)
            throw std::runtime_error("Unable to open image file");
        int magic = readInt(imageFile);
        if (magic != 2051)
            throw std::runtime_error("Invalid image file!");

        int numImages = readInt(imageFile);
        int rows = readInt(imageFile);
        int cols = readInt(imageFile);

        size_t totalSize = numImages * rows * cols;
        double *images = new double[totalSize];

        std::vector<uint8_t> buffer(rows * cols);

        for (int i = 0; i < numImages; ++i)
        {
            imageFile.read(reinterpret_cast<char *>(buffer.data()), rows * cols);
            for (int j = 0; j < rows * cols; ++j)
            {
                images[i * rows * cols + j] = static_cast<double>(buffer[j]) / 255.0; // Normalisierung
            }
        }
        return images;
    }
    std::vector<std::vector<uint8_t>> readImages()
    {
        std::ifstream imageFile(IMAGE_PATH, std::ios::binary);
        if (!imageFile)
            throw std::runtime_error("Unable to open image file");
        int magic = readInt(imageFile);
        if (magic != 2051)
            throw std::runtime_error("Invalid image file!");

        int numImages = readInt(imageFile);
        int rows = readInt(imageFile);
        int cols = readInt(imageFile);

        std::vector<std::vector<uint8_t>> images(numImages, std::vector<uint8_t>(rows * cols));
        for (int i = 0; i < numImages; ++i)
        {
            imageFile.read(reinterpret_cast<char *>(images[i].data()), rows * cols);
        }
        return images;
    }
    std::vector<uint8_t> readLabels()
    {
        std::ifstream file(LABEL_PATH, std::ios::binary);
        if (!file)
            throw std::runtime_error("Unable to open label file: labels");

        int magic = readInt(file);
        if (magic != 2049)
            throw std::runtime_error("Invalid label file!");

        int numLabels = readInt(file);

        std::vector<uint8_t> labels(numLabels);
        file.read(reinterpret_cast<char *>(labels.data()), numLabels);

        return labels;
    }
    std::vector<double> readLabelsAsDouble()
    {
        std::ifstream file(LABEL_PATH, std::ios::binary);
        if (!file)
            throw std::runtime_error("Unable to open label file");

        int magic = readInt(file);
        if (magic != 2049)
            throw std::runtime_error("Invalid label file");

        int numLabels = readInt(file);

        std::vector<uint8_t> raw(numLabels);
        file.read(reinterpret_cast<char *>(raw.data()), numLabels);

        if (!file)
            throw std::runtime_error("Failed to read labels");

        std::vector<double> labels(numLabels);
        for (int i = 0; i < numLabels; ++i)
            labels[i] = static_cast<double>(raw[i]);
        return labels;
    }
    float *loadImageAsFLoat()
    {
        std::ifstream imageFile(IMAGE_PATH, std::ios::binary);
        if (!imageFile)
            throw std::runtime_error("Unable to open image file");
        int magic = readInt(imageFile);
        if (magic != 2051)
            throw std::runtime_error("Invalid image file!");

        int numImages = readInt(imageFile);
        int rows = readInt(imageFile);
        int cols = readInt(imageFile);

        size_t totalSize = numImages * rows * cols;
        float *images = new float[totalSize];

        std::vector<uint8_t> buffer(rows * cols);

        for (int i = 0; i < numImages; ++i)
        {
            imageFile.read(reinterpret_cast<char *>(buffer.data()), rows * cols);
            for (int j = 0; j < rows * cols; ++j)
            {
                images[i * rows * cols + j] = static_cast<float>(buffer[j]) / 255.0; // Normalisierung
            }
        }
        return images;
    }
    std::vector<float> readLabelsAsFloat()
    {
        std::ifstream file(LABEL_PATH, std::ios::binary);
        if (!file)
            throw std::runtime_error("Unable to open label file");

        int magic = readInt(file);
        if (magic != 2049)
            throw std::runtime_error("Invalid label file");

        int numLabels = readInt(file);

        std::vector<uint8_t> raw(numLabels);
        file.read(reinterpret_cast<char *>(raw.data()), numLabels);

        if (!file)
            throw std::runtime_error("Failed to read labels");

        std::vector<float> labels(numLabels);
        for (int i = 0; i < numLabels; ++i)
            labels[i] = static_cast<float>(raw[i]);
        return labels;
    }

    void showImage(const std::vector<uint8_t> &pixels, int width, int height)
    {
        SDL_Init(SDL_INIT_VIDEO);

        SDL_Window *window = SDL_CreateWindow("Image", width, height, 0);
        SDL_Renderer *renderer = SDL_CreateRenderer(window, nullptr);

        // Für Graustufenbild: Wir wandeln 8-bit Graustufen in RGB um
        std::vector<uint32_t> rgbPixels(width * height);
        for (int i = 0; i < width * height; ++i)
        {
            uint8_t v = pixels[i];
            rgbPixels[i] = (0xFF << 24) | (v << 16) | (v << 8) | v; // ARGB8888
        }

        SDL_Texture *texture = SDL_CreateTexture(
            renderer,
            SDL_PIXELFORMAT_ARGB8888,
            SDL_TEXTUREACCESS_STREAMING,
            width,
            height);

        SDL_UpdateTexture(texture, nullptr, rgbPixels.data(), width * sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderTexture(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);

        SDL_Delay(2000); // 2 Sekunden anzeigen

        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }

};