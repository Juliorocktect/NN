#include "TrainingConfig.hpp"
#include "TrainingController.hpp"
#include "TrainingDataLoader.hpp"

#include <iostream>
#include <stdexcept>

int main()
{
    try
    {
        const TrainingConfig config;
        const TrainingDataLoader dataLoader;
        TrainingDataset dataset = dataLoader.load(config);

        const TrainingController trainingController;
        trainingController.run(dataset, config);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Programmfehler: " << error.what() << '\n';
        return 1;
    }
}
