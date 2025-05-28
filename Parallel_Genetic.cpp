#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <chrono>
#include <thread>
#include <mutex>

using namespace std;
using namespace chrono;

const int maxgen = 500;
const double mutations = 0.01;
const int POP_SIZE = 500;
const unsigned int NUM_THREADS = thread::hardware_concurrency();

struct BioBackPack {
    int cur_w;
    int cur_v;
};

vector<mt19937> create_generators() {
    random_device rd;
    vector<mt19937> gens;
    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        gens.push_back(mt19937(rd()));
    }
    return gens;
}

vector<int> create_individual(const vector<BioBackPack>& items, int W, mt19937& gen) {
    vector<int> individual(items.size(), 0);
    int total_weight = 0;
    uniform_real_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < items.size(); ++i) {
        if (dist(gen) < 0.5 && total_weight + items[i].cur_w <= W) {
            individual[i] = 1;
            total_weight += items[i].cur_w;
        }
    }
    return individual;
}

int fitness(const vector<int>& indiv, const vector<BioBackPack>& items, int W) {
    int total_w = 0;
    int total_v = 0;
    for(size_t i = 0; i < indiv.size(); i++) {
        if(indiv[i] == 1) {
            total_w += items[i].cur_w;
            total_v += items[i].cur_v;
        }
    }
    if(total_w > W) {
        total_v -= (total_w - W)*1000;
    }
    return total_v;
}

vector<int> selection(const vector<int>& fit_v, mt19937& gen) {
    uniform_int_distribution<int> dist(0, POP_SIZE - 1);
    vector<int> selected;
    for(int i = 0; i < POP_SIZE; i++) {
        int a = dist(gen);
        int b = dist(gen);
        selected.push_back((fit_v[a]>fit_v[b]) ? a : b);
    }
    return selected;
}

pair<vector<int>, vector<int>> baby_birth(const vector<int>& father, const vector<int>& mother, mt19937& gen) {
    uniform_int_distribution<int> dist(1, father.size() - 1);
    int point = dist(gen);
    vector<int> baby1(father.begin(), father.begin() + point);
    baby1.insert(baby1.end(), mother.begin() + point, mother.end());
    vector<int> baby2(mother.begin(), mother.begin() + point);
    baby2.insert(baby2.end(), father.begin() + point, father.end());
    return make_pair(baby1, baby2);
}

void mutation(vector<int>& indiv, mt19937& gen) {
    uniform_real_distribution<double> dist(0.0, 1.0);
    for(size_t i = 0; i < indiv.size(); i++) {
        if(dist(gen) < mutations) {
            indiv[i] = 1 - indiv[i];
        }
    }
}

vector<int> genetic_alg(const vector<BioBackPack>& items, int N, int W) {
    vector<mt19937> gens = create_generators();
    vector<vector<int>> population(POP_SIZE, vector<int>(N));
    vector<thread> init_threads;
    int chunk_size = POP_SIZE / NUM_THREADS;
    for (unsigned int t = 0; t < NUM_THREADS; ++t) {
        int start = t * chunk_size;
        int end = (t == NUM_THREADS - 1) ? POP_SIZE : start + chunk_size;
        init_threads.push_back(thread([start, end, &population, &items, W, &gens, t]() {
            for (int i = start; i < end; ++i) {
                population[i] = create_individual(items, W, gens[t]);
            }
        }));
    }
    for (size_t i = 0; i < init_threads.size(); ++i) {
        init_threads[i].join();
    }

    for(int g = 0; g < maxgen; g++) {
        vector<int> fit_v(POP_SIZE);
        vector<thread> fitness_threads;
        for (unsigned int t = 0; t < NUM_THREADS; ++t) {
            int start = t * chunk_size;
            int end = (t == NUM_THREADS - 1) ? POP_SIZE : start + chunk_size;
            fitness_threads.push_back(thread([start, end, &population, &fit_v, &items, W]() {
                for (int i = start; i < end; ++i) {
                    fit_v[i] = fitness(population[i], items, W);
                }
            }));
        }
        for (size_t i = 0; i < fitness_threads.size(); ++i) {
            fitness_threads[i].join();
        }

        vector<int> selected_ind = selection(fit_v, gens[0]);

        vector<vector<int>> new_population(POP_SIZE, vector<int>(N));

        vector<thread> breed_threads;
        for (unsigned int t = 0; t < NUM_THREADS; ++t) {
            int start = t * chunk_size;
            if (start % 2 != 0) start--;
            int end = (t == NUM_THREADS - 1) ? POP_SIZE : start + chunk_size;
            if (end % 2 != 0) end++;
            end = min(end, POP_SIZE);

            breed_threads.push_back(thread([start, end, &selected_ind, &population, &new_population, &gens, t]() {
                for (int i = start; i < end; i += 2) {
                    pair<vector<int>, vector<int>> parents = make_pair(population[selected_ind[i]], population[selected_ind[i + 1]]);
                    pair<vector<int>, vector<int>> babies = baby_birth(parents.first, parents.second, gens[t]);
                    mutation(babies.first, gens[t]);
                    mutation(babies.second, gens[t]);
                    new_population[i] = babies.first;
                    new_population[i+1] = babies.second;
                }
            }));
        }
        for (size_t i = 0; i < breed_threads.size(); ++i) {
            breed_threads[i].join();
        }
        vector<int>::iterator best_it = max_element(fit_v.begin(), fit_v.end());
        int best_index = distance(fit_v.begin(), best_it);
        new_population[0] = population[best_index];

        population = new_population;
    }

    vector<int> best_indiv;
    int best_fit = 0;
    mutex mtx;

    vector<thread> best_threads;
    for (unsigned int t = 0; t < NUM_THREADS; ++t) {
        int start = t * chunk_size;
        int end = (t == NUM_THREADS - 1) ? POP_SIZE : start + chunk_size;
        best_threads.push_back(thread([start, end, &population, &items, W, &best_fit, &best_indiv, &mtx]() {
            int local_best_fit = 0;
            vector<int> local_best_indiv;

            for (int i = start; i < end; ++i) {
                int cur_fit = fitness(population[i], items, W);
                if (cur_fit > local_best_fit) {
                    local_best_fit = cur_fit;
                    local_best_indiv = population[i];
                }
            }

            lock_guard<mutex> lock(mtx);
            if (local_best_fit > best_fit) {
                best_fit = local_best_fit;
                best_indiv = local_best_indiv;
            }
        }));
    }
    for (size_t i = 0; i < best_threads.size(); ++i) {
        best_threads[i].join();
    }

    return best_indiv;
}

int main() {
    ifstream input("ks_100_1.txt");
    if (!input) {
        cerr << "Error: could not open file" << endl;
        return 1;
    }

    int N, W;
    input >> N >> W;

    vector<BioBackPack> items(N);

    for(int i = 0; i < N; i++) {
        input >> items[i].cur_v >> items[i].cur_w;
    }

    high_resolution_clock::time_point start = high_resolution_clock::now();
    vector<int> sol = genetic_alg(items, N, W);
    high_resolution_clock::time_point end = high_resolution_clock::now();
    microseconds Comp = duration_cast<microseconds>(end - start);

    cout << fitness(sol, items, W) << " " << Comp.count() << endl;
    return 0;
}
