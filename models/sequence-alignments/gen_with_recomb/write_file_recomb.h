#include <fstream>
#include <iostream>
#include <string>

void write_recomb(std::string filename, std::string content) {
    std::ofstream file;
    file.open(filename);
    file << content;
    file.close();
}

void write_csv_recomb(std::string filename, std::vector<EXPERIMENT_RECOMB> experiments) {
    std::ofstream file;
    file.open(filename);
    file << "birth_rate,death_rate,sampling_proportion,coinfect_capacity,recomb_prop,superinfect_prop,number_of_samples,max_trys,max_time,clock_rate,evolution_model,num_sites,output_file,num_trys,count_recomb_events,final_recomb_events,index\n";
    for (int i = 0; i < experiments.size(); i++) {
        file << experiments[i].birth_rate << ",";
        file << experiments[i].death_rate << ",";
        file << experiments[i].sampling_proportion << ",";
        file << experiments[i].coinfect_capacity << ",";
        file << experiments[i].recomb_prop << ",";
        file << experiments[i].superinfect_prop << ",";
        file << experiments[i].number_of_samples << ",";
        file << experiments[i].max_trys << ",";
        file << experiments[i].max_time << ",";
        file << experiments[i].clock_rate << ",";
        file << experiments[i].evolution_model << ",";
        file << experiments[i].num_sites << ",";
        file << experiments[i].output_file << ",";
        file << experiments[i].num_trys << ",";
        file << experiments[i].count_recomb_events << ",";
        file << experiments[i].final_recomb_events << ",";
        file << experiments[i].index << "\n";
    }
    file.close();
}