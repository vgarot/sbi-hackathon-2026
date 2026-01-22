#include "experiment_design_with_recomb.h"
#include "write_file_recomb.h"
#include "tree_gen_recomb_superinfec.h"
#include "generate_msa_recomb.h"

#include <iostream>
#include <tuple>
#include <vector>
#include <string>
#include <chrono>
#include <unistd.h>
#include <filesystem>
#include <set>



int main(int argc, char *argv[])
{   
    
    if (argc != 2 && argc != 3) {
        std::cerr << "Usage: " << argv[0] << " config_file" << std::endl;
        return 1;
    }
    std::string config_file = argv[1];


    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    std::tuple<std::vector<EXPERIMENT_RECOMB>, bool, int, std::string,std::string,std::string,std::string,bool,bool> result = generate_experiments_recomb(config_file);
    std::vector<EXPERIMENT_RECOMB> experiments = std::get<0>(result);
    
    bool delete_temp = std::get<1>(result);
    int seed = std::get<2>(result);
    std::string recombinant_points_file = std::get<3>(result);
    std::string output_folder = std::get<4>(result);
    std::string tmp_dir = std::get<5>(result);
    std::string iqtree_path = std::get<6>(result);
    bool SIR = std::get<7>(result);
    bool verbose_nb_diff = std::get<8>(result); 

    try {
        std::filesystem::remove_all(output_folder);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    catch (const char* e) {
        std::cerr << "Error: " << e << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown error" << std::endl;
    }

    std::filesystem::create_directories(output_folder);
    std::filesystem::create_directories(output_folder + "/seq/");



    write_csv_recomb(output_folder + "/design.csv", experiments);

    auto parsedCsv = loadCRFs(recombinant_points_file);
    std::vector<int> weights;
    for (const auto& crf : parsedCsv) {
        weights.push_back(crf.number_of_appearances);
    }

    #pragma omp parallel for
    for (int i=0; i< experiments.size()  ; i++){
        double birth_rate = experiments[i].birth_rate;
        double death_rate = experiments[i].death_rate;
        double sampling_prop = experiments[i].sampling_proportion;
        double coinfect_capacity = experiments[i].coinfect_capacity;
        double recomb_prop = experiments[i].recomb_prop;
        double superinfect_prop = experiments[i].superinfect_prop;
        int number_of_samples = experiments[i].number_of_samples;
        int max_trys = experiments[i].max_trys;
        double max_time = experiments[i].max_time;
        double clock_rate = experiments[i].clock_rate;
        std::string evolution_model = experiments[i].evolution_model;
        int num_sites = experiments[i].num_sites;
        std::string output_file = experiments[i].output_file;
        int index = experiments[i].index;

        try {
            std::string outputfolder = tmp_dir+"/"+std::to_string(i)+"/";
            std::filesystem::create_directories(outputfolder);
            std::tuple<int,int,int> result = generate_BDSR_tree(birth_rate,
                 death_rate, 
                 sampling_prop, 
                 coinfect_capacity, 
                 recomb_prop, 
                 superinfect_prop, 
                 max_time, 
                 max_trys, 
                 number_of_samples, 
                 tmp_dir+"/"+std::to_string(i)+"/",
                 SIR
                );
            
            experiments[i].num_trys = std::get<0>(result);
            experiments[i].count_recomb_events = std::get<1>(result);
            experiments[i].final_recomb_events = std::get<2>(result);
        }
        catch (const MaxTrysReached& e) {
            experiments[i].num_trys = -1;
            std::cerr << "Error: " << e.what() << std::endl;
        }
        catch (const CapacityTooSmall& e) {
            experiments[i].num_trys = -1;
            std::cerr << "Error: " << e.what() << std::endl;
        }
        catch (const std::exception& e) {
            experiments[i].num_trys = -1;
            std::cerr << "Error: " << e.what() << std::endl;
        }
        catch (const char* e) {
            experiments[i].num_trys = -1;
            std::cerr << "Error: " << e << std::endl;
        }
        catch (...) {
            experiments[i].num_trys = -1;
            std::cerr << "Unknown error" << std::endl;
        }
    }

    write_csv_recomb(output_folder + "/design.csv", experiments);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time elapsed for tree generation: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "s" << std::endl;
    
    std::cout << "Generating the MSAs..." << std::endl;
    
    #pragma omp parallel for
    for (int i=0; i< experiments.size()  ; i++){
        if (experiments[i].num_trys == -1) {
            continue;
        }
        // std::cout << "Generating MSA for experiment " << i << std::endl;
        double clock_rate = experiments[i].clock_rate;
        std::string evol_model = experiments[i].evolution_model;
        std::string output_path = output_folder + "/seq/" + experiments[i].output_file;
        double num_sites = experiments[i].num_sites;
        std::set<std::filesystem::path> trees;
        for (auto &entry : std::filesystem::directory_iterator( tmp_dir + "/" + std::to_string(i)))
            trees.insert(entry.path());

        generate_MSA(
            trees, 
            clock_rate, 
            evol_model,
            tmp_dir+"/"+std::to_string(i)+"/", 
            output_path, 
            iqtree_path, 
            "", 
            num_sites, 
            false, 
            parsedCsv, 
            weights
        );
    }
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    std::cout << "Time elapsed for MSA generation: " << std::chrono::duration_cast<std::chrono::seconds>(end2 - end).count() << "s" << std::endl;
    if (delete_temp) {
        std::filesystem::remove_all(tmp_dir);
    }
}
