#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <functional>
#include "jsoncpp.cpp"
#include <omp.h>
struct EXPERIMENT
{
    double birth_rate;
    double death_rate;
    double sampling_proportion;
    int number_of_samples;

    int max_trys;
    double max_time;

    double clock_rate;
    std::string evolution_model;
    int num_sites;

    std::string output_file;
    int index;
    int num_trys ;

    std::string rootseq;

};

std::vector<EXPERIMENT> generate_experiments(std::string config_file) {

    std::ifstream file(config_file, std::ifstream::binary);
    Json::Value config;
    file >> config;
    int number_of_experiments = config["number_of_experiments"].asInt();
    int seed = config["seed"].asInt();
    
    // Define the generators for each thread

    std::vector<std::mt19937> generators;
    for (int i = 0; i < omp_get_max_threads(); i++) {
        generators.push_back(std::mt19937(seed + i));
    }


    // Read the config file (json)
    // Read the distributions for the parameters
    


    std::string R_0_dist_type = config["R_0"]["distribution"].asString();
    double R_0_param1 = config["R_0"]["param1"].asDouble();
    double R_0_param2 = config["R_0"]["param2"].asDouble();

    std::string infectious_time_dist_type = config["infectious_time"]["distribution"].asString();
    double infectious_time_param1 = config["infectious_time"]["param1"].asDouble();
    double infectious_time_param2 = config["infectious_time"]["param2"].asDouble();

    std::string sampling_proportion_dist_type = config["sampling_proportion"]["distribution"].asString();
    double sampling_proportion_param1 = config["sampling_proportion"]["param1"].asDouble();
    double sampling_proportion_param2 = config["sampling_proportion"]["param2"].asDouble();

    std::string number_of_samples_dist_type = config["number_of_samples"]["distribution"].asString();
    int number_of_samples_param1 = config["number_of_samples"]["param1"].asInt();
    int number_of_samples_param2 = config["number_of_samples"]["param2"].asInt();   

    std::string max_trys_dist_type = config["max_trys"]["distribution"].asString();
    int max_trys_param1 = config["max_trys"]["param1"].asInt();
    int max_trys_param2 = config["max_trys"]["param2"].asInt();

    std::string max_time_dist_type = config["max_time"]["distribution"].asString();
    double max_time_param1 = config["max_time"]["param1"].asDouble();
    double max_time_param2 = config["max_time"]["param2"].asDouble();

    std::string clock_rate_dist_type = config["clock_rate"]["distribution"].asString();
    double clock_rate_param1 = config["clock_rate"]["param1"].asDouble();
    double clock_rate_param2 = config["clock_rate"]["param2"].asDouble();

    std::string num_sites_dist_type = config["num_sites"]["distribution"].asString();
    int num_sites_param1 = config["num_sites"]["param1"].asInt();
    int num_sites_param2 = config["num_sites"]["param2"].asInt();

    std::string evolution_model = config["evolution_model"].asString();

    std::string rootseq = config["rootseq"].asString();

    

    std::function<double(int)> R_0_dis;
    if (R_0_dist_type == "lognormal") {
        std::lognormal_distribution<> dist(R_0_param1, R_0_param2);
        R_0_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    } else if (R_0_dist_type == "uniform") {
        std::uniform_real_distribution<> dist(R_0_param1, R_0_param2);
        R_0_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    } else {
        throw std::invalid_argument("Unsupported distribution type for R_0");
    }

    std::function<double(int)> infectious_time_dis;
    if (infectious_time_dist_type == "lognormal") {
        std::lognormal_distribution<> dist(infectious_time_param1, infectious_time_param2);
        infectious_time_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    } else if (infectious_time_dist_type == "uniform") {
        std::uniform_real_distribution<> dist(infectious_time_param1, infectious_time_param2);
        infectious_time_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    } else if (infectious_time_dist_type == "exponential") {
        std::exponential_distribution<> dist(1.0/infectious_time_param1);
        infectious_time_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    } else {
        throw std::invalid_argument("Unsupported distribution type for infectious_time");
    }

    std::function<double(int)> sampling_proportion_dis;
    if (sampling_proportion_dist_type == "lognormal") {
        std::lognormal_distribution<> dist(sampling_proportion_param1, sampling_proportion_param2);
        sampling_proportion_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    } else if (sampling_proportion_dist_type == "uniform") {
        std::uniform_real_distribution<> dist(sampling_proportion_param1, sampling_proportion_param2);
        sampling_proportion_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    } else if (sampling_proportion_dist_type == "powered10uniform") {
        std::uniform_real_distribution<> dist(sampling_proportion_param1, sampling_proportion_param2);
        sampling_proportion_dis = [dist, &generators](int thread_num) mutable { return std::pow(10,dist(generators[thread_num])); };
    } else {
        throw std::invalid_argument("Unsupported distribution type for sampling_proportion");
    }

    std::function<int(int)> number_of_samples_dis;
    if (number_of_samples_dist_type == "uniform") {
        std::uniform_int_distribution<> dist(number_of_samples_param1, number_of_samples_param2);
        number_of_samples_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    } else {
        throw std::invalid_argument("Unsupported distribution type for number_of_samples");
    }

    std::function<int(int)> max_trys_dis;
    if (max_trys_dist_type == "uniform") {
        std::uniform_int_distribution<> dist(max_trys_param1, max_trys_param2);
        max_trys_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    } else {
        throw std::invalid_argument("Unsupported distribution type for max_trys");
    }

    std::function<double(int)> max_time_dis;
    if (max_time_dist_type == "uniform") {
        std::uniform_real_distribution<> dist(max_time_param1, max_time_param2);
        max_time_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    } else {
        throw std::invalid_argument("Unsupported distribution type for max_time");
    }

    std::function<double(int)> clock_rate_dis;
    if (clock_rate_dist_type == "lognormal") {
        std::lognormal_distribution<> dist(clock_rate_param1, clock_rate_param2);
        clock_rate_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    } else if (clock_rate_dist_type == "uniform") {
        std::uniform_real_distribution<> dist(clock_rate_param1, clock_rate_param2);
        clock_rate_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    } else if (clock_rate_dist_type == "powered10uniform") {
        std::uniform_real_distribution<> dist(clock_rate_param1, clock_rate_param2);
        clock_rate_dis = [dist, &generators](int thread_num) mutable { return std::pow(10,dist(generators[thread_num])); };
    }else {
        throw std::invalid_argument("Unsupported distribution type for clock_rate");
    }

    std::function<int(int)> num_sites_dis;
    if (num_sites_dist_type == "uniform") {
        std::uniform_int_distribution<> dist(num_sites_param1, num_sites_param2);
        num_sites_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    } else {
        throw std::invalid_argument("Unsupported distribution type for num_sites");
    }

    std::vector<EXPERIMENT> experiments;
    
    // ///////////////
    // // CHANGE HERE THE DISTRIBUTIONS FOR THE PARAMETERS
    // std::lognormal_distribution<> R_0_dis(0.8,0.4);
    // std::lognormal_distribution<> infectious_time_dis(1.2,0.5);
    // std::uniform_real_distribution<> sampling_proportion_dis(-2,0);
    // std::uniform_int_distribution<> number_of_samples_dis(20,100);
    // std::uniform_int_distribution<> max_trys_dis(200,200);
    // std::uniform_real_distribution<> max_time_dis(10000.0,10000.0);
    // std::uniform_real_distribution<> clock_rate_dis(-4,-2); // 0.00014 for 1.4 x 10^-4 subst/site/28days
    // std::uniform_int_distribution<> num_sites_dis(500,500);
    // std::string evolution_model = "JC";


    /////////////
    #pragma omp parallel 
    {
        std::vector<EXPERIMENT> local_experiments;
        #pragma omp for nowait schedule(static)
        for (int i = 0; i < number_of_experiments; i++) {

            EXPERIMENT experiment;

            // output_file format = BD__birth_rate__death_rate__sampling_proportion.newick
            int current_thread = omp_get_thread_num();

            double R_0 = R_0_dis(current_thread);
            double infectious_time = infectious_time_dis(current_thread);
            double sampling_proportion = sampling_proportion_dis(current_thread);
            int number_of_samples = number_of_samples_dis(current_thread);
            int max_trys = max_trys_dis(current_thread);
            double max_time = max_time_dis(current_thread);
            double clock_rate = clock_rate_dis(current_thread);
            int num_sites = num_sites_dis(current_thread);

            double birth_rate = R_0 / infectious_time;
            double death_rate = 1.0 / infectious_time;
            
            experiment.birth_rate = birth_rate;
            experiment.death_rate = death_rate;
            experiment.sampling_proportion = sampling_proportion;
            experiment.number_of_samples = number_of_samples;
            experiment.max_trys = max_trys;
            experiment.max_time = max_time;
            experiment.clock_rate = clock_rate;
            experiment.evolution_model = evolution_model;
            experiment.num_sites = num_sites;
            experiment.rootseq = rootseq;
            experiment.output_file = "BD__" + std::to_string(birth_rate) + "__" + std::to_string(death_rate) + "__" + std::to_string(experiment.sampling_proportion) + ".newick";
            experiment.index = i;

            local_experiments.push_back(experiment);
        }
        #pragma omp for schedule(static) ordered
        for (int i = 0; i<omp_get_num_threads(); i++) {
            #pragma omp ordered
            experiments.insert(experiments.end(), local_experiments.begin(), local_experiments.end());
        }
    }

    return experiments;
}