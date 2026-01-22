#include <filesystem>
#include <set>
#include <stdexcept>
#include <tuple>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <map>



class CRF {
    public:
        std::string name;
        int number_of_appearances;
        std::vector<int> breakpoints;
    
        CRF (std::string name,int number_of_appearances, std::vector<int> breakpoints) {
            this->name = name;
            this->breakpoints = breakpoints;
            this->number_of_appearances = number_of_appearances;
        };
};

std::vector<CRF> loadCRFs(const std::string& filename) {
    std::ifstream  data(filename);
    std::string line;
    std::vector<CRF> parsedCsv;
    std::vector<int> weights;

    while(std::getline(data,line))
    {
        // Discard header line
        if (line.rfind("name", 0) == 0) {
            continue;
        }
        std::stringstream lineStream(line);
        std::string name;
        std::string cell;
        int number_of_appearances;
        std::vector<int> breakpoints;
        int idx = 0;
        while (std::getline(lineStream, cell, '\t')) {
            if (idx == 0) {
                name = cell;
            }
            else if (idx == 1) {
                number_of_appearances = std::stoi(cell);
                weights.push_back(number_of_appearances);
            }
            else if (idx == 2) {
                // Assuming breakpoints are separated by - in the cell
                std::stringstream breakpointsStream(cell);
                std::string breakpoint;
                while (std::getline(breakpointsStream, breakpoint, '-')) {
                    breakpoints.push_back(std::stoi(breakpoint));
                }
            }
            else {
                throw std::invalid_argument("Too many columns in recombination points file");
            }
            idx++;
        }

        CRF parsedRow(name, number_of_appearances, breakpoints);


        parsedCsv.push_back(parsedRow);
    }
    data.close();
    return parsedCsv;
};



class NotGenerated : public std::exception {
    public:
        NotGenerated() {};
        const char* what() const throw() {
            return "Not generated";
        }
};


std::string recombine(
    std::string seq1,
    std::string seq2, 
    const std::vector<CRF>& parsedCsv, 
    const std::vector<int>& weights,
    bool verbose_nb_diff = false
)
{ 
   
    if (verbose_nb_diff) {
        int differences = 0;
        for (int i = 0; i < seq1.size(); i++) {
            if (seq1.at(i) != seq2.at(i)) {
                differences ++;
            }
        }
        std::cout << "Number of differences between sequences: " << differences << std::endl;
    }
    std::random_device rd;
    std::mt19937 gen(rd());


    std::discrete_distribution<> dis(weights.begin(), weights.end());
    int index = dis(gen);
    CRF selected = parsedCsv.at(index);
    std::bernoulli_distribution coinflip(0.5);
    int value = coinflip(gen);
    
    std::string recombined_seq = "";
    int previous_bp = 0;
    

    for (int i = 0; i < selected.breakpoints.size(); i++) {
        int bp = selected.breakpoints.at(i);
        if ((i + value) % 2 == 0) {
            recombined_seq += seq1.substr(previous_bp, bp - previous_bp);
        }
        else {
            recombined_seq += seq2.substr(previous_bp, bp - previous_bp);
        }
        previous_bp = bp;
    }
    // add the last segment
    if ((selected.breakpoints.size() + value) % 2 == 0) {
        recombined_seq += seq1.substr(previous_bp, seq1.size() - previous_bp);
    }
    else {
        recombined_seq += seq2.substr(previous_bp, seq2.size() - previous_bp);
    }
    return recombined_seq;
}
    




std::map<std::string,std::string> find_recomb(
    std::filesystem::path path,
    bool verbose = false
) { 
    if (verbose) {
        std::cout << "In " << path << std::endl;
    }

    std::map<std::string,std::string> result;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::invalid_argument("File not found");
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind(">recomb", 0) == 0) {
            std::string recomb = line;
            std::string seq;
            std::getline(file, seq);
            result[recomb.substr(1, recomb.rfind("_")+1)] = seq;
        }
    }
    file.close();
    return result;
}

std::vector<std::filesystem::path> sort_files(std::set<std::filesystem::path> files) {
    std::vector<std::filesystem::path> result;
    std::vector<std::pair<int, std::filesystem::path>> indexes;
    for (auto &filename : files) {
        result.push_back(filename);
        if (filename.u8string().find("base") != std::string::npos) {
            indexes.push_back(std::make_pair(0, filename));
        }
        else {
            int index = std::stoi(filename.u8string().substr(filename.u8string().find_last_of("b") + 1, filename.u8string().find_last_of("_") -1 - filename.generic_string().find_last_of("b")));
            indexes.push_back(std::make_pair(index, filename));
        }
    }
    std::sort(indexes.begin(), indexes.end());

    result.clear();
    for (int i = 0; i < indexes.size(); i++) {
        result.push_back(indexes[i].second);
    }
    return result;
}
    

void generate_MSA(
    std::set<std::filesystem::path> trees,
    double clock_rate,
    std::string evol_model,
    std::string tmp_dir,
    std::string output_path,
    std::string iqtree_path,
    std::string root_seq_path = "",
    int num_sites = -1,
    bool verbose = false,
    const std::vector<CRF>& parsedCsv = std::vector<CRF>(),
    const std::vector<int>& weights = std::vector<int>(),
    bool verbose_nb_diff = false
) { 
    if (verbose) {
        std::cout << "Generating MSA" << std::endl;
    }
    std::vector<std::filesystem::path> sorted_files = sort_files(trees);


    std::string command_init = iqtree_path;

    command_init = command_init + " -m " + evol_model;
    command_init = command_init + " --branch-scale " + std::to_string(clock_rate);
    command_init = command_init + " -af fasta ";

    std::string command = command_init; 
    std::vector<std::filesystem::path> files_to_merge;

    std::map<std::string,std::string> recombinants;

    for (int i = 0; i < sorted_files.size(); i++) {
        if (verbose) {
            std::cout << "Processing " << sorted_files[i] << std::endl;
        }
        if (i == 0) {
            
            if (root_seq_path != "") {
                command = command + " --root-seq " + root_seq_path;
            }
            else {
                if (num_sites != -1) {
                    command = command + " --length " + std::to_string(num_sites);
                }
                else throw std::invalid_argument("root_seq_path or num_sites must be provided");
            }

            command = command + " --alisim " + tmp_dir + "base";
            command = command + " -t " + sorted_files[i].u8string();
            if (!verbose) {
                command = command + " > /dev/null";
            }
            system(command.c_str());
            files_to_merge.push_back(tmp_dir + "base.fa");
            recombinants = find_recomb(tmp_dir + "base.fa", verbose);
            continue;
        }
        std::string recomb = sorted_files[i].u8string().substr(sorted_files[i].u8string().find_last_of("/") + 1, sorted_files[i].u8string().find_last_of(".") - 1 - sorted_files[i].u8string().find_last_of("/"));

        std::string recomb1 = recomb.substr(0, recomb.find("__"));
        std::string recomb2 = recomb.substr(recomb.rfind("__") + 2, recomb.size() - recomb.find("__") - 2);

        if (verbose) {
            std::cout << "Recomb1: " << recomb1 << std::endl;
            std::cout << "seq1: " << recombinants[recomb1] << std::endl;
            std::cout << "Recomb2: " << recomb2 << std::endl;
            std::cout << "seq2: " << recombinants[recomb2] << std::endl;
        }

        std::string seq_recombined = recombine(recombinants[recomb1], recombinants[recomb2], parsedCsv, weights, verbose_nb_diff);

        if (verbose) {
            std::cout << "Recombined seq: " << std::endl;
            std::cout << seq_recombined << std::endl;

            std::cout << "from: " << recomb1 << " and " << recomb2 << std::endl;
            std::cout << std::endl << std::endl;
        }
        

        std::ofstream file(tmp_dir + recomb + ".fa");
        file << ">recomb" << std::endl;
        file << seq_recombined << std::endl;
        file.close();

        std::ifstream tree(sorted_files[i]);
        std::string line;
        std::getline(tree, line);
        line = line.substr(0, line.size() - 1);
        line = "(rootseq:0.0," + line + "):0.0;";
        std::ofstream new_tree(tmp_dir + recomb + ".newick");
        new_tree << line;
        new_tree.close();


        command = command_init;
        command = command + " --root-seq " + tmp_dir + recomb + ".fa,recomb";
        command = command + " --alisim " + tmp_dir + recomb;
        command = command + " -t " + tmp_dir + recomb + ".newick";
        if (!verbose) {
            command = command + " > /dev/null";
        }
        system(command.c_str());
        files_to_merge.push_back(tmp_dir + recomb + ".fa");

        std::map<std::string,std::string> new_recombinants = find_recomb(tmp_dir + recomb + ".fa", verbose);
        recombinants.insert(new_recombinants.begin(), new_recombinants.end());        
    }
    std::ofstream final_msa(output_path);
    for (int i=0 ; i < files_to_merge.size(); i++) {
        std::ifstream file(files_to_merge[i]);
        std::string line;
        while (std::getline(file, line)) {
            if (line.rfind(">recomb", 0) == 0 or line.rfind(">rootseq", 0) == 0){
                std::getline(file, line);
                continue;
            }
            final_msa << line << std::endl;
        }
        file.close();
    }
    final_msa.close();
    return;

}