#include "write_file.h"
#include "tree_gen.h"
#include <iostream>
#include <tuple>
#include <vector>
#include <string>
#include <chrono>
#include <unistd.h>

std::tuple<double, double> transform_birth_death_sampling(double birth_rate, double death_rate, double sampling_proportion) {
    double alpha = birth_rate - death_rate;
    double beta = birth_rate * death_rate * sampling_proportion;

    double new_birth_rate = (alpha + std::sqrt(alpha*alpha + 4*beta)) / 2;
    double new_death_rate = new_birth_rate - alpha;

    return std::make_tuple(new_birth_rate, new_death_rate);
};


int main(int argc, char *argv[])
{
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: " << argv[0] << " config_file output_folder (seed)" << std::endl;
        return 1;
    }
    std::string config_file = argv[1];
    std::string output_folder = argv[2];

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    std::vector<EXPERIMENT> experiments = generate_experiments(config_file);

    write_csv(output_folder + "/design.csv", experiments);

    #pragma omp parallel for
    for (int i = 0; i < experiments.size(); i++) {
        double birth_rate = experiments[i].birth_rate;
        double death_rate = experiments[i].death_rate;
        double sampling_proportion = experiments[i].sampling_proportion;
        int number_of_samples = experiments[i].number_of_samples;
        int max_trys = experiments[i].max_trys;
        double max_time = experiments[i].max_time;
        double clock_rate = experiments[i].clock_rate;
        std::string evolution_model = experiments[i].evolution_model;
        int num_sites = experiments[i].num_sites;
        std::string output_file = experiments[i].output_file;
        int index = experiments[i].index;

        std::tuple<double, double> new_rates = transform_birth_death_sampling(birth_rate, death_rate, sampling_proportion);
        double new_birth_rate = std::get<0>(new_rates);
        double new_death_rate = std::get<1>(new_rates);

        try {
            
            std::pair<std::string,int> result = generate_BDS_tree(new_birth_rate, new_death_rate, max_time, max_trys, number_of_samples);
            experiments[i].num_trys = result.second;
            output_file = output_folder + "/trees/" + output_file;
            write(output_file, result.first + ";");

        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            continue;
        }
        catch (const char* e) {
            std::cerr << "Error: " << e << std::endl;
            continue;
        }
        catch (...) {
            std::cerr << "Unknown error" << std::endl;
            continue;
        }
        
    }

    write_csv(output_folder + "/design.csv", experiments);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time elapsed for tree generation: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "s" << std::endl;
    
}
