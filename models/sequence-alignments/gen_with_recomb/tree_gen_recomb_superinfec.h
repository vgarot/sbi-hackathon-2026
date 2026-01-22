#include <vector> 
#include <map>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>

class MaxTrysReached : public std::exception {
    public:
        MaxTrysReached() {};
        const char* what() const throw() {
            return "Max trys reached";
        }
};

class CapacityTooSmall : public std::exception {
    public:
        CapacityTooSmall() {};
        const char* what() const throw() {
            return "Population capacity too small";
        }
};


class Node {
    public:
        Node* parent;
        double dist; //dist to his parent
        double dist_to_start;
        std::vector<Node*> children;
        bool is_leaf;
        bool is_root;
        std::string name;
        std::vector<Node*> parents;
        bool to_delete = false;

        Node (double dist) {
            this->dist = dist;
            this->is_leaf = false;
            this->children = std::vector<Node*>();
            this->parent = nullptr;
            this->is_root = false;
            this->name = "";
        };

        // only 2 children possible

        void add_childs(Node* n1, Node* n2) {
            if (is_leaf) {
                throw std::invalid_argument("Leaf can't have children");
            } else {
                if (children.size() > 0) {
                    throw std::invalid_argument("Node already have 2 children");
                }
                children.push_back(n1);
                children.push_back(n2);
                n1->parent = this;
                n2->parent = this;
            }
        };
        Node* get_child(int c) {
            if (is_leaf) {
                throw std::invalid_argument("Leaf can't have children");
            } else {
                if (c > 1) {
                    throw std::invalid_argument("c must be 0 or 1");
                } else {
                    return children[c];
                }
            }
        };  


        void remove() {
            
            if (is_root) {
                if (parents == std::vector<Node*>()) {
                    this->to_delete = true;
                    return;
                }
                else {
                    Node* first_parent = parents[0];
                    Node* second_parent = parents[1];
                    first_parent->remove();
                    second_parent->remove();
                    this->to_delete = true;
                    return;
                }
            }
            if (parent->children.size() == 1) {
                parent->remove();
                delete this;
                return;
            }
            Node* first_child = parent->get_child(0);
            int c;
            if (first_child == this) {
                c = 0;
            } else {
                c = 1;
            }
            Node* child = parent->get_child(1-c);
            child->dist += parent->dist;
            Node* grand_parent = parent->parent;
            if (grand_parent->children[0] == parent) {
                grand_parent->children[0] = child;
            } else {
                grand_parent->children[1] = child;
            }
            child->parent = grand_parent;
            delete parent;
            delete this;
        };

        std::string write_to_newick() {
            std::string returning;
            if (is_leaf) {
                returning = name + ':' + std::to_string(dist);
                delete this;
                return returning;
            } else {
                if (children.size() == 1) {
                    returning = '(' + children[0]->write_to_newick() + ')' + ':' + std::to_string(dist);
                    delete this;
                    return returning;
                } else {
                if (children.size()==0) {
                    delete this;
                    return "";
                }
                returning = '(' + children[0]->write_to_newick() + ',' + children[1]->write_to_newick() + ')' + ':' + std::to_string(dist);
                delete this;
                return returning;}
            }
        };

};


class Rates {
    public:
        double trans_rate;
        double remove_rate;
        double coinfect_prop;

        double superinfect_prop;
        double recomb_prop;

        double sum_rate;

        Rates (double x, double y,double z, double w, double t) {
            birth_rate = x;
            death_rate = y;
            pop_capacity = z;
            recomb_prop = w;
            superinfect_prop = t;
        };
        void update_rates (int number_infectious) {
            trans_rate = birth_rate * number_infectious;
            remove_rate = death_rate * number_infectious;;
            if (pop_capacity <=1) {
                throw CapacityTooSmall();
            }
            coinfect_prop = (number_infectious-1) / pop_capacity ;
            
            sum_rate = trans_rate + remove_rate ;
        }
        void decrease_capacity() {
            pop_capacity -= 1;
        }

    private:
        double birth_rate;
        double death_rate;
        double pop_capacity;
};


std::tuple<int,int,int> generate_BDSR_tree(
    double birth_rate,
    double death_rate,
    double sampling_prop,
    double coinfect_capacity,
    double recomb_prop,
    double superinfect_prop,
    double max_time,
    int max_trys,
    int wanted_samplesd,
    std::string tmp_dir,
    bool removal = false
) {
    

    Rates rates(birth_rate, death_rate,coinfect_capacity,recomb_prop,superinfect_prop);

    int number_infectious;
    bool stop = false;
    int trys = 0;
    double actual_time;
    int sampled;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    Node* root;
    int total;
    std::vector<Node*> infectious_people;
    std::vector<Node*> roots;
    int count_recomb_events = 0;

    while (!stop && trys < max_trys) {
        if (trys>0) {

            for (int i=0; i<infectious_people.size(); i++) {
                infectious_people[i]->remove();
            }
            int deleted = 0;
            int inisize = roots.size();


            

            for (int i=0; i<inisize; i++) {
                
                if (roots.at(inisize - i -1)->to_delete){
                    delete roots.at(inisize - i -1);
                    roots.erase(roots.begin() + inisize - i -1);
                    deleted ++;
                }
            }

            if (roots.size()>0) {

                for (int i=0; i<roots.size()-1; i++) {
                    roots.at(roots.size() - i -1)->children.at(0)->write_to_newick();
                    delete roots.at(roots.size() - i -1);
                }
                roots.at(0)->write_to_newick();
            }
        }
        total = 0;
        number_infectious = 1;
        sampled = 0;
        actual_time = 0.0;
        count_recomb_events = 0;
        root = new Node(0.0);
        root->dist_to_start = 0.0;
        root->is_root = true;
        Node* first_infectious = new Node(0.0);
        first_infectious->dist_to_start = 0.0;
        root->children.push_back(first_infectious);
        first_infectious->parent = root;
        
        roots.clear();
        roots.push_back(root);

        infectious_people.clear();
        infectious_people.push_back(first_infectious);

        while (actual_time < max_time && number_infectious>0 && sampled < wanted_samplesd) {
            try {
                rates.update_rates(number_infectious);
            }
            catch (CapacityTooSmall& e) {
                break;
            }
            total ++;
            // time before next event
            double u = dis(gen);
            double waiting_time = -std::log(u) / rates.sum_rate;
            actual_time += waiting_time;

            // chose type of event : transmission, removal, coinfection or recombination
            double event = dis(gen);
            event *= rates.sum_rate;

            int index_affected = int(dis(gen) * number_infectious);

            Node* affected = infectious_people.at(index_affected);


            

            if (event<rates.trans_rate) { //transmission

                if (event < rates.trans_rate * (1 - rates.coinfect_prop)) {

                    infectious_people.erase(infectious_people.begin() + index_affected);
                    affected->dist = actual_time - affected->parent->dist_to_start;
                    affected->dist_to_start = actual_time;

                    number_infectious ++;
                    Node* donnor = new Node(0.0);
                    Node* receiver = new Node(0.0);
                    donnor->dist_to_start = actual_time;
                    receiver->dist_to_start = actual_time;
                    affected->add_childs(donnor, receiver);
                    infectious_people.push_back(donnor);
                    infectious_people.push_back(receiver);

                }
                else if (event < rates.trans_rate * (1 - rates.coinfect_prop * rates.superinfect_prop)) { 
                } // coinf but without reaching

                else if (event < rates.trans_rate * (1 - rates.coinfect_prop * rates.superinfect_prop * rates.recomb_prop)) {
                    //coinf without recomb (just taking over)
                    infectious_people.erase(infectious_people.begin() + index_affected);
                    affected->dist = actual_time - affected->parent->dist_to_start;
                    affected->dist_to_start = actual_time;

                    Node* donnor = new Node(0.0);
                    Node* receiver = new Node(0.0);

                    int index_coinf = int(dis(gen) * (number_infectious-1));
                    Node* coinfect = infectious_people.at(index_coinf);
                    infectious_people.erase(infectious_people.begin() + index_coinf);
                    coinfect->remove();

                    
                    donnor->dist_to_start = actual_time;
                    receiver->dist_to_start = actual_time;
                    affected->add_childs(donnor, receiver);
                    infectious_people.push_back(donnor);
                    infectious_people.push_back(receiver);
                }
                else {
                    // recomb
                    count_recomb_events ++;
                    infectious_people.erase(infectious_people.begin() + index_affected);

                    affected->dist = actual_time - affected->parent->dist_to_start;
                    affected->dist_to_start = actual_time;
                    int index_coinf = int(dis(gen) * (number_infectious-1));

                    Node* coinfect = infectious_people.at(index_coinf);

                    infectious_people.erase(infectious_people.begin() + index_coinf);

                    coinfect->dist = actual_time - coinfect->parent->dist_to_start;
                    coinfect->dist_to_start = actual_time;

                    coinfect->is_leaf = true;
                    coinfect->name = "recomb" + std::to_string(total) + "_2";

                    Node* indiv_1 = new Node(0.0);
                    Node* checkpoint = new Node(0.0);

                    indiv_1->dist_to_start = actual_time;
                    checkpoint->dist_to_start = actual_time;

                    checkpoint->is_leaf = true;
                    checkpoint->name = "recomb" + std::to_string(total) + "_1";

                    affected->add_childs(indiv_1, checkpoint);
                    infectious_people.push_back(indiv_1);


                    
                    Node* new_root = new Node(0.0);
                    new_root->dist_to_start = actual_time;
                    new_root->is_root = true;


                    Node* indiv_2 = new Node(0.0);
                    new_root->children.push_back(indiv_2);
                    indiv_2->parent = new_root;
                    new_root->parents.push_back(checkpoint);
                    new_root->parents.push_back(coinfect);


                    infectious_people.push_back(indiv_2);

                    roots.push_back(new_root);
                }
            }
            else { //removal
                double sampling = dis(gen);
                if (sampling < sampling_prop) {
                    if (removal) {
                        rates.decrease_capacity();
                    }
                    infectious_people.erase(infectious_people.begin() + index_affected);
                    affected->dist = actual_time - affected->parent->dist_to_start;
                    affected->dist_to_start = actual_time;

                    number_infectious --;
                    sampled ++;
                    affected->is_leaf = true;
                    affected->name = "n" + std::to_string(sampled) + "|" + std::to_string(affected->dist_to_start);
                }
                else {
                    infectious_people.erase(infectious_people.begin() + index_affected);
                    number_infectious --;
                    affected->remove();
                }
            }
                

            
        };
        trys ++;
        if (sampled == wanted_samplesd) {
            stop = true;
        }
    }
    if (trys == max_trys) {
        throw MaxTrysReached();
    }
    for (int i=0; i<infectious_people.size(); i++) {
            infectious_people[i]->remove();
        }
   
    int deleted = 0;
    int inisize = roots.size();

    for (int i=0; i<inisize; i++) {
        
        if (roots.at(inisize - i -1)->to_delete){
            delete roots.at(inisize - i -1);
            roots.erase(roots.begin() + inisize - i -1);
            deleted ++;
        }
    }
    int final_recomb_events = roots.size() -1;
    
    for (int i=0; i<roots.size()-1; i++) {
        std::string parent1 = roots.at(roots.size() - i -1)->parents.at(0)->name;
        std::string parent2 = roots.at(roots.size() - i -1)->parents.at(1)->name;
        std::string outfile = tmp_dir + parent1 + "__" + parent2 + ".newick";
        std::string newick =  roots.at(roots.size() - i -1)->children.at(0)->write_to_newick();
        write_recomb(outfile, newick + ";");
        delete roots.at(roots.size() - i -1);
    }
    std::string newick = roots.at(0)->children.at(0)->write_to_newick();
    delete roots.at(0);
    write_recomb(tmp_dir+"/base.newick", newick + ";");
    return std::make_tuple(trys, count_recomb_events, final_recomb_events);

};