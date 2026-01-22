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
#include <variant>


class InputParser{
    public:
        InputParser (int &argc, char **argv){
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }

        const std::string& getCmdOption(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }

        bool cmdOptionExists(const std::string &option) const{
            return std::find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }
    private:
        std::vector <std::string> tokens;
};


class MissingArgument : public std::exception {
    public:
        MissingArgument(const std::string& arg) : argument(arg) {};
        const char* what() const throw() {
            return ("Missing required argument: " + argument).c_str();
        }
    private:
        std::string argument;
};
 
std::variant<int, std::string, double> getCmdOptionOrThrow(const InputParser& input, const std::string& option, const std::string& type) {
    const std::string& value = input.getCmdOption(option);
    if (value.empty()) {
        throw MissingArgument(option);
    }
    if (type == "int") {
        return std::stoi(value);
    } else if (type == "double") {
        return std::stod(value);
    } else if (type == "string") {
        return value;
    } else {
        throw std::invalid_argument("Unsupported type for command line argument");
    }
}

void generate_all(
    double birth_rate,
    double death_rate,
    double sampling_prop,
    double coinfect_capacity,
    double recomb_prop,
    double superinfect_prop,
    double max_time,
    int max_trys,
    int number_of_samples,
    std::string tmp_dir,
    bool SIR,
    std::string evol_model,
    double clock_rate,
    std::string iqtree_path,
    std::string output_path,
    int num_sites,
    std::string recombinant_points_file,
    bool verbose_nb_diff,
    bool delete_files,
    std::string root_seq_path = ""
){
    tmp_dir = tmp_dir + "/";
    if (std::filesystem::exists(tmp_dir)) {
        std::filesystem::remove_all(tmp_dir);
    }
    std::filesystem::create_directories(tmp_dir);

    std::tuple<int,int,int> result = generate_BDSR_tree(
                birth_rate,
                death_rate, 
                sampling_prop, 
                coinfect_capacity, 
                recomb_prop, 
                superinfect_prop, 
                max_time, 
                max_time, 
                number_of_samples, 
                tmp_dir,
                SIR
                );
    
    int num_trys = std::get<0>(result);
    int count_recomb_events = std::get<1>(result);
    int final_recomb_events = std::get<2>(result);

    std::set<std::filesystem::path> tree_files;
    for (const auto& entry : std::filesystem::directory_iterator(tmp_dir)) {
        tree_files.insert(entry.path());
    }

    auto parsedCsv = loadCRFs(recombinant_points_file);
    std::vector<int> weights;
    for (const auto& crf : parsedCsv) {
        weights.push_back(crf.number_of_appearances);
    }
    

    if (num_trys != -1) {
        generate_MSA(
            tree_files,
            clock_rate,
            evol_model,
            tmp_dir,
            output_path,
            iqtree_path,
            root_seq_path,
            num_sites,
            false,
            parsedCsv,
            weights,
            verbose_nb_diff
        );
    }

    if (!delete_files)
        return ;
    std::filesystem::remove_all(tmp_dir);
    return ;

}




int main(int argc, char **argv){
    InputParser input(argc, argv);
    if(input.cmdOptionExists("-h")){
        std::cout << 
        "Usage : " << argv[0] << " [OPTIONS]\n" <<
        "Options:\n" <<
        "  --birth_rate <double>                Birth rate\n" <<
        "  --death_rate <double>                Death rate\n" <<
        "  --sampling_prop <double>             Sampling proportion\n" <<
        "  --coinfect_capacity <double>         Coinfection capacity\n" <<
        "  --recomb_prop <double>               Recombination proportion\n" <<
        "  --superinfect_prop <double>          Superinfection proportion\n" <<
        "  --max_time <double>                  Maximum time for simulation\n" <<
        "  --max_trys <int>                     Maximum number of tries for simulation\n" <<
        "  --number_of_samples <int>            Number of samples to generate\n" <<
        "  --tmp_dir <string>                   Temporary directory for intermediate files\n" <<
        "  --evol_model <string>                Evolutionary model for IQ-TREE\n" <<
        "  --clock_rate <double>                Clock rate for IQ-TREE\n" <<
        "  --iqtree_path <string>               Path to IQ-TREE executable\n" <<
        "  --output_path <string>               Output path for the generated MSA\n" <<
        "  --num_sites <int>                    Number of sites in the sequences\n" <<
        "  --recombinant_points_file <string>   CSV file with recombinant points information\n" <<
        "  [--root_seq_path <string>]           (Optional) Path to root sequence file\n" <<
        "  [--SIR]                              Use SIR model (default is SIS)\n" <<
        "  [--verbose_nb_diff]                  Verbose number of differences between recombined sequences\n" <<
        "  [--no_delete_temp]                   Do not delete temporary files\n" <<
        std::endl;
        return 0;
    }

    double birth_rate = std::get<double>(getCmdOptionOrThrow(input, "--birth_rate", "double"));
    double death_rate = std::get<double>(getCmdOptionOrThrow(input, "--death_rate", "double"));
    double sampling_prop = std::get<double>(getCmdOptionOrThrow(input, "--sampling_prop", "double"));
    double coinfect_capacity = std::get<double>(getCmdOptionOrThrow(input, "--coinfect_capacity", "double"));
    double recomb_prop = std::get<double>(getCmdOptionOrThrow(input, "--recomb_prop", "double"));
    double superinfect_prop = std::get<double>(getCmdOptionOrThrow(input, "--superinfect_prop", "double"));
    double max_time = std::get<double>(getCmdOptionOrThrow(input, "--max_time", "double"));
    int max_trys = std::get<int>(getCmdOptionOrThrow(input, "--max_trys", "int"));
    int number_of_samples = std::get<int>(getCmdOptionOrThrow(input, "--number_of_samples", "int"));
    std::string tmp_dir = std::get<std::string>(getCmdOptionOrThrow(input, "--tmp_dir", "string"));

    bool SIR = false;
    if (input.cmdOptionExists("--SIR")) {
        SIR = true;
    }

    std::string evol_model = std::get<std::string>(getCmdOptionOrThrow(input, "--evol_model", "string"));
    double clock_rate = std::get<double>(getCmdOptionOrThrow(input, "--clock_rate", "double"));
    std::string iqtree_path = std::get<std::string>(getCmdOptionOrThrow(input, "--iqtree_path", "string"));
    std::string output_path = std::get<std::string>(getCmdOptionOrThrow(input, "--output_path", "string"));
    int num_sites = std::get<int>(getCmdOptionOrThrow(input, "--num_sites", "int"));
    std::string root_seq_path = "";
    if (input.cmdOptionExists("--root_seq_path")) {
        root_seq_path = std::get<std::string>(getCmdOptionOrThrow(input, "--root_seq_path", "string"));
    }
    std::string recombinant_points_file = std::get<std::string>(getCmdOptionOrThrow(input, "--recombinant_points_file", "string"));
    bool verbose_nb_diff = false;
    if (input.cmdOptionExists("--verbose_nb_diff")) {
        verbose_nb_diff = true;
    }
    bool delete_files = true;
    if (input.cmdOptionExists("--no_delete_temp")) {
        delete_files = false;
    }

    generate_all(
        birth_rate,
        death_rate,
        sampling_prop,
        coinfect_capacity,
        recomb_prop,
        superinfect_prop,
        max_time,
        max_trys,
        number_of_samples,
        tmp_dir,
        SIR,
        evol_model,
        clock_rate,
        iqtree_path,
        output_path,
        num_sites,
        recombinant_points_file,
        verbose_nb_diff,
        delete_files,
        root_seq_path
    );
}