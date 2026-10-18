#include <VNN.h>
#include <ImageLoading.h>
#include <memory>
#include <vector>
#include <iostream>
#include <stdexcept>

int main()
{
    try
    {
        constexpr size_t trainingCount = 40000;
        std::vector<float> labels = ImagePreProcessor::readLabelsAsFloat();
        if (labels.size() < trainingCount)
        {
            throw std::runtime_error("Weniger als 40000 Labels geladen");
        }

        std::unique_ptr<float[]> images(ImagePreProcessor::loadImageAsFLoat(trainingCount));
        NNG network(trainingCount);
        network.initilizeYMatrix(labels.data());
        network.setInputData(images.get());
        network.run(20000);

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Programmfehler: " << error.what() << '\n';
        return 1;
    }

}