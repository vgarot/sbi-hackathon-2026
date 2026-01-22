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



class Node {
    public:
        Node* parent;
        double dist; //dist to his parent
        double dist_to_start;
        std::vector<Node*> children;
        bool is_leaf; // only for root
        std::string name;

        Node (double dist) {
            this->dist = dist;
            this->is_leaf = false;
            this->children = std::vector<Node*>();
            this->parent = nullptr;

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
            if (parent == nullptr) {
                throw std::invalid_argument("Root can't be removed");
            }
            if (parent->children.size() == 1) {
                throw std::invalid_argument("Node can't be removed, it's the only child of his parent");
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
        double sum_rate; 
        Rates (double x, double y) {
            birth_rate = x;
            death_rate = y;
        };
        void update_rates (int number_infectious) {
            trans_rate = birth_rate * number_infectious;
            remove_rate = death_rate * number_infectious;
            sum_rate = trans_rate + remove_rate;
        }

        void transform_rate () {
            // Transformer les rates vers du (lambda, mu , 1) 
        }
    private:
        double birth_rate;
        double death_rate;
};


std::pair<std::string,int> generate_BDS_tree(
    double birth_rate,
    double death_rate,
    double max_time,
    int max_trys,
    int wanted_samplesd
) {
    

    Rates rates(birth_rate, death_rate);

    int number_infectious;
    bool stop = false;
    int trys = 0;
    double actual_time;
    int sampled;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    Node root(0.0);
    std::vector<Node*> infectious_people;
    int total;

    while (!stop && trys < max_trys) {
        total = 0;
        number_infectious = 1;
        sampled = 0;
        actual_time = 0.0;
        root = Node(0.0);
        root.dist_to_start = 0.0;
        Node* first_infectious = new Node(0.0);
        first_infectious->dist_to_start = 0.0;
        root.children.push_back(first_infectious);
        first_infectious->parent = &root;
        
        infectious_people.clear();
        infectious_people.push_back(first_infectious);

        while (actual_time < max_time && number_infectious>0 && sampled < wanted_samplesd) {
            rates.update_rates(number_infectious);
            
            // time before next event
            double u = dis(gen);
            double waiting_time = -std::log(u) / rates.sum_rate;
            actual_time += waiting_time;

            // chose affected one
            int index_affected = int(dis(gen) * number_infectious);
            Node* affected = infectious_people[index_affected];

            infectious_people.erase(infectious_people.begin() + index_affected);

            affected->dist = actual_time - affected->parent->dist_to_start;
            affected->dist_to_start = actual_time;

            // kind of event : transmission or removal
            double event = dis(gen);
            double threshold = rates.trans_rate / rates.sum_rate;

            if (event < threshold) { //transmission
                total ++;
                number_infectious ++;
                Node* donnor = new Node(0.0);
                Node* receiver = new Node(0.0);
                donnor->dist_to_start = actual_time;
                receiver->dist_to_start = actual_time;
                affected->add_childs(donnor, receiver);
                infectious_people.push_back(donnor);
                infectious_people.push_back(receiver);

            } else { //removal
                number_infectious --;
                sampled ++;
                affected->is_leaf = true;
                affected->name = "n" + std::to_string(sampled) + "|" + std::to_string(affected->dist_to_start);
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
    std::string newick = root.get_child(0)->write_to_newick();

    return std::make_pair(newick, trys);


};