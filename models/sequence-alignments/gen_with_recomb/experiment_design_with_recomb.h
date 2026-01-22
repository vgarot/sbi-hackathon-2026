#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <functional>
#include <fstream>
#include "jsoncpp.cpp"
#include <omp.h>

struct EXPERIMENT_RECOMB
{
    double birth_rate;
    double death_rate;
    double sampling_proportion;
    double coinfect_capacity;
    double recomb_prop;
    double superinfect_prop;

    int number_of_samples;

    int max_trys;
    double max_time;

    double clock_rate;
    std::string evolution_model;
    int num_sites;

    std::string output_file;
    int index;
    int num_trys = 0;

    int count_recomb_events = 0 ;
    int final_recomb_events = 0;

};

std::tuple<std::vector<EXPERIMENT_RECOMB>, bool, int, std::string,std::string,std::string,std::string,bool,bool> generate_experiments_recomb(
    std::string config_file
) {
    std::ifstream file(config_file, std::ifstream::binary);
    Json::Value config;
    file >> config;
    int number_of_experiments = config["number_of_experiments"].asInt();
    int seed = config["seed"].asInt();
    bool delete_temp = config["delete_temp_files"].asBool();
    std::string recombinant_points_file = config["recombinant_points_file"].asString();
    std::string output_directory = config["output_directory"].asString();
    std::string tmp_dir = config["tmp_dir"].asString();
    std::string iqtree_path = config["iqtree_path"].asString();
    bool SIR = config["capacity_decrease"].asBool();
    bool verbose_nb_diff = config.get("verbose_nb_diff", false).asBool();

    // Define the generators for each thread

    std::vector<std::mt19937> generators;
    for (int i = 0; i < omp_get_max_threads(); i++) {
        generators.push_back(std::mt19937(seed + i));
    };

    std::function<std::tuple<std::string,double,double>(Json::Value, std::string)> extract_from =
        [](Json::Value config_file,
        std::string key
    )
    {
        std::string distribution = config_file[key]["distribution"].asString();
        double param1 = config_file[key]["param1"].asDouble();
        double param2 = config_file[key]["param2"].asDouble();
        return std::make_tuple(distribution, param1, param2);
    };

    std::function<std::function<double(int)>(std::tuple<std::string,double,double>)> define_dis = [&generators](
        std::tuple<std::string,double,double> dist_info
    ) -> std::function<double(int)> {  
        std::string dist_type = std::get<0>(dist_info);
        double param1 = std::get<1>(dist_info);
        double param2 = std::get<2>(dist_info);

        if (dist_type == "lognormal") {
            return [param1, param2, &generators](int thread_num) mutable {
                std::lognormal_distribution<> dist(param1, param2);
                return dist(generators[thread_num]);
            };
        } else if (dist_type == "uniform") {
            return [param1, param2, &generators](int thread_num) mutable {
                std::uniform_real_distribution<> dist(param1, param2);
                return dist(generators[thread_num]);
            };
        } else if (dist_type == "powered10uniform") {
            return [param1, param2, &generators](int thread_num) mutable {
                std::uniform_real_distribution<> dist(param1, param2);
                return std::pow(10, dist(generators[thread_num]));
            };
        } else {
            throw std::invalid_argument("Unsupported distribution type: " + dist_type);
        }
    };

    std::function<double(int)> R_0_dis = define_dis(extract_from(config, "R_0"));;
    std::function<double(int)> infectious_time_dis = define_dis(extract_from(config, "infectious_time"));;
    std::function<double(int)> sampling_proportion_dis = define_dis(extract_from(config, "sampling_proportion"));;
    std::function<double(int)> coinfect_capacity_dis = define_dis(extract_from(config, "coinfect_capacity"));;
    std::function<double(int)> recomb_prop_dis = define_dis(extract_from(config, "recomb_prop"));;
    std::function<double(int)> superinfect_prop_dis = define_dis(extract_from(config, "superinfect_prop"));;
    std::function<int(int)> number_of_samples_dis = define_dis(extract_from(config, "number_of_samples"));;
    std::function<int(int)> max_trys_dis = define_dis(extract_from(config, "max_trys"));;
    std::function<double(int)> max_time_dis = define_dis(extract_from(config, "max_time"));;
    std::function<double(int)> clock_rate_dis = define_dis(extract_from(config, "clock_rate"));;
    std::function<int(int)> num_sites_dis = define_dis(extract_from(config, "num_sites"));;


    // std::string R_0_dist_type = config["R_0"]["distribution"].asString();
    // double R_0_param1 = config["R_0"]["param1"].asDouble();
    // double R_0_param2 = config["R_0"]["param2"].asDouble();

    // std::string infectious_time_dist_type = config["infectious_time"]["distribution"].asString();
    // double infectious_time_param1 = config["infectious_time"]["param1"].asDouble();
    // double infectious_time_param2 = config["infectious_time"]["param2"].asDouble();

    // std::string sampling_proportion_dist_type = config["sampling_proportion"]["distribution"].asString();
    // double sampling_proportion_param1 = config["sampling_proportion"]["param1"].asDouble();
    // double sampling_proportion_param2 = config["sampling_proportion"]["param2"].asDouble();

    // std::string coinfect_capacity_dist_type = config["coinfect_capacity"]["distribution"].asString();
    // double coinfect_capacity_param1 = config["coinfect_capacity"]["param1"].asDouble();
    // double coinfect_capacity_param2 = config["coinfect_capacity"]["param2"].asDouble();

    // std::string recomb_prop_dist_type = config["recomb_prop"]["distribution"].asString();
    // double recomb_prop_param1 = config["recomb_prop"]["param1"].asDouble();
    // double recomb_prop_param2 = config["recomb_prop"]["param2"].asDouble();

    // std::string superinfect_prop_dist_type = config["superinfect_prop"]["distribution"].asString();
    // double superinfect_prop_param1 = config["superinfect_prop"]["param1"].asDouble();
    // double superinfect_prop_param2 = config["superinfect_prop"]["param2"].asDouble();

    // std::string number_of_samples_dist_type = config["number_of_samples"]["distribution"].asString();
    // int number_of_samples_param1 = config["number_of_samples"]["param1"].asInt();
    // int number_of_samples_param2 = config["number_of_samples"]["param2"].asInt();   

    // std::string max_trys_dist_type = config["max_trys"]["distribution"].asString();
    // int max_trys_param1 = config["max_trys"]["param1"].asInt();
    // int max_trys_param2 = config["max_trys"]["param2"].asInt();

    // std::string max_time_dist_type = config["max_time"]["distribution"].asString();
    // double max_time_param1 = config["max_time"]["param1"].asDouble();
    // double max_time_param2 = config["max_time"]["param2"].asDouble();

    // std::string clock_rate_dist_type = config["clock_rate"]["distribution"].asString();
    // double clock_rate_param1 = config["clock_rate"]["param1"].asDouble();
    // double clock_rate_param2 = config["clock_rate"]["param2"].asDouble();

    // std::string num_sites_dist_type = config["num_sites"]["distribution"].asString();
    // int num_sites_param1 = config["num_sites"]["param1"].asInt();
    // int num_sites_param2 = config["num_sites"]["param2"].asInt();

    std::string evolution_model = config["evolution_model"].asString();

    std::string rootseq = config["rootseq"].asString();

    

    // std::function<double(int)> R_0_dis;
    // if (R_0_dist_type == "lognormal") {
    //     std::lognormal_distribution<> dist(R_0_param1, R_0_param2);
    //     R_0_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    // } else if (R_0_dist_type == "uniform") {
    //     std::uniform_real_distribution<> dist(R_0_param1, R_0_param2);
    //     R_0_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    // } else {
    //     throw std::invalid_argument("Unsupported distribution type for R_0");
    // }

    // std::function<double(int)> infectious_time_dis;
    // if (infectious_time_dist_type == "lognormal") {
    //     std::lognormal_distribution<> dist(infectious_time_param1, infectious_time_param2);
    //     infectious_time_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    // } else if (infectious_time_dist_type == "uniform") {
    //     std::uniform_real_distribution<> dist(infectious_time_param1, infectious_time_param2);
    //     infectious_time_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    // } else {
    //     throw std::invalid_argument("Unsupported distribution type for infectious_time");
    // }

    // std::function<double(int)> sampling_proportion_dis;
    // if (sampling_proportion_dist_type == "lognormal") {
    //     std::lognormal_distribution<> dist(sampling_proportion_param1, sampling_proportion_param2);
    //     sampling_proportion_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    // } else if (sampling_proportion_dist_type == "uniform") {
    //     std::uniform_real_distribution<> dist(sampling_proportion_param1, sampling_proportion_param2);
    //     sampling_proportion_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    // } else if (sampling_proportion_dist_type == "powered10uniform") {
    //     std::uniform_real_distribution<> dist(sampling_proportion_param1, sampling_proportion_param2);
    //     sampling_proportion_dis = [dist, &generators](int thread_num) mutable { return std::pow(10,dist(generators[thread_num])); };
    // } else {
    //     throw std::invalid_argument("Unsupported distribution type for sampling_proportion");
    // }

    // std::function<double(int)> coinfect_capacity_dis;
    // if (coinfect_capacity_dist_type == "uniform") {
    //     std::uniform_real_distribution<> dist(coinfect_capacity_param1, coinfect_capacity_param2);
    //     coinfect_capacity_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    // } else {
    //     throw std::invalid_argument("Unsupported distribution type for coinfect_capacity");
    // }
    // std::function<double(int)> recomb_prop_dis;
    // if (recomb_prop_dist_type == "uniform") {
    //     std::uniform_real_distribution<> dist(recomb_prop_param1, recomb_prop_param2);
    //     recomb_prop_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    // } else {
    //     throw std::invalid_argument("Unsupported distribution type for recomb_prop");
    // }

    // std::function<double(int)> superinfect_prop_dis;
    // if (superinfect_prop_dist_type == "uniform") {
    //     std::uniform_real_distribution<> dist(superinfect_prop_param1, superinfect_prop_param2);
    //     superinfect_prop_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    // } else {
    //     throw std::invalid_argument("Unsupported distribution type for superinfect_prop");
    // }

    // std::function<int(int)> number_of_samples_dis;
    // if (number_of_samples_dist_type == "uniform") {
    //     std::uniform_int_distribution<> dist(number_of_samples_param1, number_of_samples_param2);
    //     number_of_samples_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    // } else {
    //     throw std::invalid_argument("Unsupported distribution type for number_of_samples");
    // }

    // std::function<int(int)> max_trys_dis;
    // if (max_trys_dist_type == "uniform") {
    //     std::uniform_int_distribution<> dist(max_trys_param1, max_trys_param2);
    //     max_trys_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    // } else {
    //     throw std::invalid_argument("Unsupported distribution type for max_trys");
    // }

    // std::function<double(int)> max_time_dis;
    // if (max_time_dist_type == "uniform") {
    //     std::uniform_real_distribution<> dist(max_time_param1, max_time_param2);
    //     max_time_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    // } else {
    //     throw std::invalid_argument("Unsupported distribution type for max_time");
    // }

    // std::function<double(int)> clock_rate_dis;
    // if (clock_rate_dist_type == "lognormal") {
    //     std::lognormal_distribution<> dist(clock_rate_param1, clock_rate_param2);
    //     clock_rate_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    // } else if (clock_rate_dist_type == "uniform") {
    //     std::uniform_real_distribution<> dist(clock_rate_param1, clock_rate_param2);
    //     clock_rate_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    // } else if (clock_rate_dist_type == "powered10uniform") {
    //     std::uniform_real_distribution<> dist(clock_rate_param1, clock_rate_param2);
    //     clock_rate_dis = [dist, &generators](int thread_num) mutable { return std::pow(10,dist(generators[thread_num])); };
    // }else {
    //     throw std::invalid_argument("Unsupported distribution type for clock_rate");
    // }

    // std::function<int(int)> num_sites_dis;
    // if (num_sites_dist_type == "uniform") {
    //     std::uniform_int_distribution<> dist(num_sites_param1, num_sites_param2);
    //     num_sites_dis = [dist, &generators](int thread_num) mutable { return dist(generators[thread_num]); };
    // } else {
    //     throw std::invalid_argument("Unsupported distribution type for num_sites");
    // }


    std::vector<EXPERIMENT_RECOMB> experiments;

    #pragma omp parallel
    {
        std::vector<EXPERIMENT_RECOMB> local_experiments;
        #pragma omp for nowait schedule(static)
        for (int i = 0; i < number_of_experiments; i++) {
            EXPERIMENT_RECOMB experiment;
            int current_thread = omp_get_thread_num();
            double R_0 = R_0_dis(current_thread);
            double infectious_time = infectious_time_dis(current_thread);
            double sampling_proportion = sampling_proportion_dis(current_thread);
            double coinfect_capacity = coinfect_capacity_dis(current_thread);
            double recomb_prop = recomb_prop_dis(current_thread);
            double superinfect_prop = superinfect_prop_dis(current_thread);
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
            experiment.coinfect_capacity = coinfect_capacity;
            experiment.recomb_prop = recomb_prop;
            experiment.superinfect_prop = superinfect_prop;
            experiment.number_of_samples = number_of_samples;
            experiment.max_trys = max_trys;
            experiment.max_time = max_time;
            experiment.clock_rate = clock_rate;
            experiment.evolution_model = evolution_model;
            experiment.num_sites = num_sites;
            experiment.output_file = "BDSR__" + std::to_string(birth_rate) + "__" + std::to_string(death_rate) + "__" + std::to_string(sampling_proportion) + ".fa";
            experiment.index = i;

            local_experiments.push_back(experiment);
        }
        #pragma omp for schedule(static) ordered
        for (int i = 0; i<omp_get_num_threads(); i++) {
            #pragma omp ordered
            experiments.insert(experiments.end(), local_experiments.begin(), local_experiments.end());
        }
    }
    return std::make_tuple(experiments, delete_temp, seed, recombinant_points_file, output_directory,tmp_dir, iqtree_path, SIR, verbose_nb_diff);
}