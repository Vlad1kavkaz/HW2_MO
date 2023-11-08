#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>

const int POPULATION_SIZE = 4;
const int NUM_GENERATIONS = 3000;
const double MUTATION_RATE = 0.1;
const double MIN_X = -2.0;
const double MAX_X = 2.0;
const double MIN_Y = -2.0;
const double MAX_Y = 2.0;
const int MAX_EXECUTION_TIME_SECONDS = 15;
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

std::vector<Individual> select_parents(const std::vector<Individual> &population) {
    std::vector<Individual> parents = population;
    std::sort(parents.begin(), parents.end(), [](const Individual &a, const Individual &b) {
        return a.fitness > b.fitness;
    });
    return parents;
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

Individual mutate(const Individual &individual) {
    auto currentTime = std::chrono::system_clock::now().time_since_epoch();
    unsigned seed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime).count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> mutation_prob(0.0, 1.0);
    std::uniform_real_distribution<double> mutation_value(-2.0, 2.0);

    double x = individual.x;
    double y = individual.y;

    if (mutation_prob(gen) < MUTATION_RATE) {
        x = mutation_value(gen);
    }

    if (mutation_prob(gen) < MUTATION_RATE) {
        y = mutation_value(gen);
    }

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
    double tolerance = 0.01; // Порог для сходимости

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
    std::vector<int> iterationsToPrint = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
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

    for (int generation = 0; generation < NUM_GENERATIONS; generation++) {
        population = evaluate_population(population);
        std::vector<Individual> parents = select_parents(population);
        printToTxt(population, generation);

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