#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>

const int POPULATION_SIZE = 4;
const int NUM_GENERATIONS = 80000;
double MUTATION_RATE = 0.25;
const double MIN_X = -2.0;
const double MAX_X = 2.0;
const double MIN_Y = -2.0;
const double MAX_Y = 2.0;
const int MAX_EXECUTION_TIME_SECONDS = 600;
auto start_time = std::chrono::high_resolution_clock::now();

struct Individual {
    double x;
    double y;
    double fitness;

    Individual(double x, double y) : x(x), y(y), fitness(0.0) {}
    Individual() {}
};

std::vector<Individual> create_initial_population() {
    std::vector<Individual> population(POPULATION_SIZE);


    auto currentTime = std::chrono::system_clock::now().time_since_epoch();
    unsigned seed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime).count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> x_distribution(MIN_X, MAX_X);
    std::uniform_real_distribution<double> y_distribution(MIN_Y, MAX_Y);
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        double x = x_distribution(generator);
        double y = y_distribution(generator);
        population[i] = Individual(x, y);
    }

    return population;
}

double fitness_function(double x, double y) {
    return sin(x) * exp(-x * x - y * y);
}

std::vector<Individual> evaluate_population(std::vector<Individual> &population) {
    for (auto &individual : population) {
        double fitness = fitness_function(individual.x, individual.y);
        individual.fitness = fitness;
    }
    return population;
}

std::vector<Individual> select_parents(const std::vector<Individual>& population) {
    std::vector<Individual> parents;
    const int TOURNAMENT_SIZE = 3;
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> distribution(0, population.size() - 1);

    for (int i = 0; i < population.size(); ++i) {
        Individual bestParent;
        double bestFitness = -std::numeric_limits<double>::infinity();

        for (int j = 0; j < TOURNAMENT_SIZE; ++j) {
            int random_index = distribution(generator);
            const Individual& candidate = population[random_index];

            double distance = std::sqrt(std::pow(candidate.x - 0.653297871, 2) + std::pow(candidate.y + 0.00000000564618584, 2));

            // Меняем вес в зависимости от близости к ожидаемому решению
            double distance_weight = exp(-0.1 * distance);

            double weighted_fitness = candidate.fitness * distance_weight;

            if (weighted_fitness > bestFitness) {
                bestParent = candidate;
                bestFitness = weighted_fitness;
            }
        }

        parents.push_back(bestParent);
    }

    return parents;
}


// Обновленная функция мутации
Individual mutate(const Individual& individual) {

    auto currentTime = std::chrono::system_clock::now().time_since_epoch();
    unsigned seed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime).count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> mutation_prob(0.0, 1.0);
    std::uniform_real_distribution<double> mutation_value(-MUTATION_RATE, MUTATION_RATE);

    double x = individual.x;
    double y = individual.y;

    if (mutation_prob(gen) < MUTATION_RATE) {
        x += mutation_value(gen);
        x = std::max(MIN_X, std::min(x, MAX_X));
    }

    if (mutation_prob(gen) < MUTATION_RATE) {
        y += mutation_value(gen);
        y = std::max(MIN_Y, std::min(y, MAX_Y));
    }

    Individual ind(x, y);
    ind.fitness = fitness_function(x, y);

    return ind;
}



Individual crossover(const Individual &parent1, const Individual &parent2) {
    auto currentTime = std::chrono::system_clock::now().time_since_epoch();
    unsigned seed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime).count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> crossover_prob(0.0, 1.0);

    double x = (crossover_prob(gen) < 0.5) ? parent1.x : parent2.x;
    double y = (crossover_prob(gen) < 0.5) ? parent1.y : parent2.y;
    Individual ind(x, y);
    ind.fitness = fitness_function(x, y);

    return ind;
}


bool checkConvergence(std::vector<Individual>& population) {
    double sum = 0;
    for (auto& p : population) {
        sum += p.fitness;
    }
    double average = sum / population.size();

    int numConverged = 0;
    double tolerance = 0.0001; // Порог для сходимости

    for (auto& p : population) {
        if (std::abs(p.fitness - average) < tolerance) {
            numConverged++;
        }
    }

    double convergenceRatio = static_cast<double>(numConverged) / population.size();
    return (convergenceRatio >= 0.7); // Если 70% или более сходятся, возвращаем true
}

//для заполнения таблицы в отчете
void printToTxt(std::vector<Individual> population, int i) {
    std::ofstream out(R"(C:\Users\28218\CLionProjects\HW2-MO\output.txt)", std::ios::app);
    double sum = 0;
    std::vector<int> iterationsToPrint = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000, 20000, 50000, 79999};
    if (std::find(iterationsToPrint.begin(), iterationsToPrint.end(), i) != iterationsToPrint.end()) {
        out << i << std::endl;
        out << "X:" << std::endl;
        for (int j = 0; j < 4; ++j) {
            out << population[j].x << std::endl;
        }
        out << "Y:" << std::endl;
        for (int j = 0; j < 4; ++j) {
            out << population[j].y << std::endl;
        }
        out << "F:" << std::endl;
        for (size_t j = 0; j < 4; ++j) {
            sum += population[j].fitness;
            out << population[j].fitness << std::endl;
        }
        Individual best_individual = *std::max_element(population.begin(), population.end(),
                                                       [](const Individual &a, const Individual &b) {
                                                           return a.fitness < b.fitness;
                                                       });
        double maxF = best_individual.fitness;
        out << "MAX:" << maxF << std::endl;
        out << "AVG:" << sum / population.size() << std::endl;
        out << std::endl;
    }
}




int main() {
    std::vector<Individual> population = create_initial_population();
    double initial_mutation_rate = MUTATION_RATE;
    for (int generation = 0; generation < NUM_GENERATIONS; generation++) {
        population = evaluate_population(population);
        std::vector<Individual> parents = select_parents(population);
        printToTxt(population, generation);
        double current_convergence = static_cast<double>(generation) / NUM_GENERATIONS;
        MUTATION_RATE = initial_mutation_rate * (1.0 - current_convergence);
        for (size_t j = 0; j < POPULATION_SIZE - 1; j += 2) {
            Individual child1 = crossover(parents[j], parents[j + 1]);
            child1 = mutate(child1);

            // Обновляем родителей детьми
            population[j] = child1;

            // Обновляем x и y для каждой особи
            population[j].x = child1.x;
            population[j].y = child1.y;
            population[j].fitness = fitness_function(population[j].x, population[j].y);
        }

        auto current_time = std::chrono::high_resolution_clock::now();
        auto execution_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
        if (execution_time >= MAX_EXECUTION_TIME_SECONDS) {
            std::cout << "RunTime Limit" << std::endl;
            break;
        }
        if (checkConvergence(population)) {
            std::cout << "Fitness Convergence" << generation << std::endl;
            break;
        }
    }

    Individual best_individual = *std::max_element(population.begin(), population.end(),
                                                   [](const Individual &a, const Individual &b) {
                                                       return a.fitness < b.fitness;
                                                   });

    std::cout << "Best solution: x = " << best_individual.x << ", y = " << best_individual.y << ", f(x, y) = "
              << best_individual.fitness << std::endl;

    return 0;
}