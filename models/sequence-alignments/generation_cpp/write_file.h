#include <fstream>
#include <iostream>
#include <string>
#include "experiment_design.h"

void write(std::string filename, std::string content) {
    std::ofstream file;
    file.open(filename);
    file << content;
    file.close();
}

void write_csv(std::string filename, std::vector<EXPERIMENT> experiments) {
    std::ofstream file;
    file.open(filename);
    file << "birth_rate;death_rate;sampling_proportion;number_of_samples;max_trys;max_time;clock_rate;evolution_model;num_sites;output_file;num_trys;index;rootseq\n";
    for (int i = 0; i < experiments.size(); i++) {
        file << experiments[i].birth_rate << ";";
        file << experiments[i].death_rate << ";";
        file << experiments[i].sampling_proportion << ";";
        file << experiments[i].number_of_samples << ";";
        file << experiments[i].max_trys << ";";
        file << experiments[i].max_time << ";";
        file << experiments[i].clock_rate << ";";
        file << experiments[i].evolution_model << ";";
        file << experiments[i].num_sites << ";";
        file << experiments[i].output_file << ";";
        file << experiments[i].num_trys << ";";
        file << experiments[i].index << ";";
        file << experiments[i].rootseq << "\n";
    }
    file.close();
}