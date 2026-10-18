#include "ImageLoading.h"

#include <gtk/gtk.h>
#include <stdexcept>

namespace
{
struct ImageDisplay
{
    GMainLoop *loop;
    guint timeoutSourceId = 0;
};

void drawImage(GtkDrawingArea *, cairo_t *context, int width, int height, gpointer userData)
{
    auto *surface = static_cast<cairo_surface_t *>(userData);
    const double scaleX = static_cast<double>(width) / cairo_image_surface_get_width(surface);
    const double scaleY = static_cast<double>(height) / cairo_image_surface_get_height(surface);
    cairo_save(context);
    cairo_scale(context, scaleX, scaleY);
    cairo_set_source_surface(context, surface, 0.0, 0.0);
    cairo_paint(context);
    cairo_restore(context);
}

gboolean closeImageWindow(GtkWindow *, gpointer userData)
{
    auto *display = static_cast<ImageDisplay *>(userData);
    if (display->timeoutSourceId != 0)
    {
        g_source_remove(display->timeoutSourceId);
        display->timeoutSourceId = 0;
    }
    g_main_loop_quit(display->loop);
    return FALSE;
}

gboolean finishImageDisplay(gpointer userData)
{
    auto *display = static_cast<ImageDisplay *>(userData);
    display->timeoutSourceId = 0;
    g_main_loop_quit(display->loop);
    return G_SOURCE_REMOVE;
}
}

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

    float *loadImageAsFLoat(size_t imageCount)
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

        if (numImages <= 0 || rows <= 0 || cols <= 0)
            throw std::runtime_error("Invalid image metadata: dimensions must be positive");

        const size_t requestedImages = (imageCount == 0)
                                           ? static_cast<size_t>(numImages)
                                           : std::min<size_t>(imageCount, static_cast<size_t>(numImages));

        const size_t imageStride = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        const size_t totalSize = requestedImages * imageStride;
        float *images = new float[totalSize]();
        std::vector<uint8_t> buffer(imageStride);

        for (size_t i = 0; i < requestedImages; ++i)
        {
            if (!imageFile.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(imageStride)))
            {
                delete[] images;
                throw std::runtime_error("Failed to read image block");
            }

            for (size_t j = 0; j < imageStride; ++j)
            {
                images[j * requestedImages + i] = static_cast<float>(buffer[j]) / 255.0f;
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

    float *loadImageTestData(size_t imageCount)
    {
        std::ifstream imageFile(IMAGE_TEST_PATH, std::ios::binary);
        if (!imageFile)
            throw std::runtime_error("Unable to open image file");

        int magic = readInt(imageFile);
        if (magic != 2051)
            throw std::runtime_error("Invalid image file!");

        int numImages = readInt(imageFile);
        int rows = readInt(imageFile);
        int cols = readInt(imageFile);

        if (numImages <= 0 || rows <= 0 || cols <= 0)
            throw std::runtime_error("Invalid image metadata: dimensions must be positive");

        const size_t requestedImages = (imageCount == 0)
                                           ? static_cast<size_t>(numImages)
                                           : std::min<size_t>(imageCount, static_cast<size_t>(numImages));

        const size_t imageStride = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        const size_t totalSize = requestedImages * imageStride;
        float *images = new float[totalSize]();
        std::vector<uint8_t> buffer(imageStride);

        for (size_t i = 0; i < requestedImages; ++i)
        {
            if (!imageFile.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(imageStride)))
            {
                delete[] images;
                throw std::runtime_error("Failed to read image block");
            }

            for (size_t j = 0; j < imageStride; ++j)
            {
                images[j * requestedImages + i] = static_cast<float>(buffer[j]) / 255.0f;
            }
        }

        return images;
    }
    std::vector<float> readTestLabels()
    {
        std::ifstream file(LABEL_TEST_PATH, std::ios::binary);
        if (!file)
            throw std::runtime_error("Unable to open test label file");

        int magic = readInt(file);
        if (magic != 2049)
            throw std::runtime_error("Invalid test label file");

        int numLabels = readInt(file);
        if (numLabels <= 0)
            throw std::runtime_error("Invalid test label count");

        std::vector<uint8_t> raw(numLabels);
        file.read(reinterpret_cast<char *>(raw.data()), numLabels);

        if (!file)
            throw std::runtime_error("Failed to read test labels");

        std::vector<float> labels(numLabels);
        for (int i = 0; i < numLabels; ++i)
            labels[i] = static_cast<float>(raw[i]);
        return labels;
    }

    void showImage(const std::vector<uint8_t> &pixels, int width, int height)
    {
        if (width <= 0 || height <= 0 ||
            pixels.size() != static_cast<size_t>(width) * static_cast<size_t>(height))
            throw std::invalid_argument("Invalid image data or dimensions");

        gtk_init();
        std::vector<uint32_t> rgbPixels(pixels.size());
        for (size_t i = 0; i < pixels.size(); ++i)
        {
            const uint32_t value = pixels[i];
            rgbPixels[i] = 0xFF000000U | (value << 16) | (value << 8) | value;
        }

        cairo_surface_t *surface = cairo_image_surface_create_for_data(
            reinterpret_cast<unsigned char *>(rgbPixels.data()),
            CAIRO_FORMAT_RGB24,
            width,
            height,
            width * static_cast<int>(sizeof(uint32_t)));
        if (cairo_surface_status(surface) != CAIRO_STATUS_SUCCESS)
        {
            const cairo_status_t status = cairo_surface_status(surface);
            cairo_surface_destroy(surface);
            throw std::runtime_error(cairo_status_to_string(status));
        }

        GtkWindow *window = GTK_WINDOW(gtk_window_new());
        g_object_ref_sink(window);
        gtk_window_set_title(window, "Image");
        gtk_window_set_default_size(window, width, height);
        GtkWidget *drawingArea = gtk_drawing_area_new();
        gtk_drawing_area_set_content_width(GTK_DRAWING_AREA(drawingArea), width);
        gtk_drawing_area_set_content_height(GTK_DRAWING_AREA(drawingArea), height);
        gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(drawingArea), drawImage, surface, nullptr);
        gtk_window_set_child(window, drawingArea);

        ImageDisplay display{g_main_loop_new(nullptr, FALSE)};
        g_signal_connect(window, "close-request", G_CALLBACK(closeImageWindow), &display);
        display.timeoutSourceId = g_timeout_add_seconds(2, finishImageDisplay, &display);
        gtk_window_present(window);
        g_main_loop_run(display.loop);
        if (display.timeoutSourceId != 0)
            g_source_remove(display.timeoutSourceId);
        gtk_window_destroy(window);
        g_object_unref(window);
        g_main_loop_unref(display.loop);
        cairo_surface_destroy(surface);
    }

    void showImage(const float *pixels, int width, int height)
    {
        if (pixels == nullptr || width <= 0 || height <= 0)
            throw std::invalid_argument("Invalid image data or dimensions");

        std::vector<uint8_t> bytePixels(static_cast<size_t>(width) * static_cast<size_t>(height));
        for (size_t i = 0; i < bytePixels.size(); ++i)
        {
            float value = pixels[i];
            if (value < 0.0f)
                value = 0.0f;
            else if (value > 1.0f)
                value = 1.0f;

            bytePixels[i] = static_cast<uint8_t>(value * 255.0f);
        }

        showImage(bytePixels, width, height);
    }

};