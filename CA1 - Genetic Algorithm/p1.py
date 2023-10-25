from typing import NamedTuple
import random
import copy
import matplotlib.pyplot as plt
import numpy as np


class Input(NamedTuple):
    ex_len: int
    coefficients: list[int]
    goal: dict[int, int]

class Chromosome:
    def __init__(self, genes: list[int], fitness: float):
        self.genes = genes
        self.fitness = fitness
        
    @classmethod
    def gen_random(cls, input: Input):
        genes: list[int] = [None for _ in range(input.ex_len)]
        for i in range(0, input.ex_len):
            gene = random.choice(input.coefficients)
            if(i == input.ex_len -1):
                while(gene == 0):
                    gene = random.choice(input.coefficients)
            genes[i] = gene
        x = cls(genes, 0)
        return x

    def eval_exp(self, x):
        res = 0
        for i in range(0, len(self.genes)):
            res += self.genes[i]*pow(x, i)
        return res

    def eval_diff_exps(self, input: Input):
        sum_diff = 0
        for x in input.goal:
            sum_diff += abs(input.goal[x] - self.eval_exp(x))
        return sum_diff

    def calc_fitness(self, input: Input):
        diff = self.eval_diff_exps(input)
        self.fitness = float(1 /(1+diff))

    def mutate(self, input: Input,prob_mut_gene: float):
        for i in range(0,len(self.genes)):
            if random.random() < prob_mut_gene:
                gene = random.choice(input.coefficients)
                self.genes[i]= gene
                self.makenonzero(input)

    def makenonzero(self, input):
        while self.genes[len(self.genes)-1] == 0 :
            self.genes[len(self.genes)-1] = random.choice(input.coefficients)

class Genetic:
    def __init__(self, input: Input, prob_xover, prob_mut_chromosome, prob_mut_gene, prob_carry, population_size, max_generation):
        self.input = input
        self.prob_xover = prob_xover
        self.prob_mut_chromosome = prob_mut_chromosome
        self.prob_mut_gene = prob_mut_gene
        self.prob_carry = prob_carry
        self.population_size = population_size
        self.max_generation = max_generation
        
    def generate_population(self):
        population = [Chromosome.gen_random(self.input) for _ in range(self.population_size)]
        return population

    def select_one(self, population: list[Chromosome]):
        index = 0
        r = random.random()
        while r > 0 and index < len(population):
            r -= population[index].fitness
            index += 1
        index -= 1
        return population[index]

    def mating_pool(self, population: list[Chromosome]):
        pop = copy.deepcopy(population)
        sum_fitness = sum(chromosome.fitness for chromosome in population)
        for chromosome in pop:
            chromosome.fitness =  chromosome.fitness / sum_fitness
        pop = sorted(pop,key=lambda x: -x.fitness)
        new_generation_parents = [copy.deepcopy(self.select_one(pop)) for _ in population]
        return new_generation_parents

    def crossover(self, population: list[Chromosome]):
        new_generation = []
        for i in range(0, len(population), 2):
            children1 = population[i]
            children2 = population[i+1]
            if random.random() <= self.prob_xover:
                point = random.randint(0, self.input.ex_len-1)
                children1.genes[:point], children2.genes[point:] = children2.genes[:point], children1.genes[point:]
            new_generation.append(children1)
            new_generation.append(children2)
        return new_generation

    def mutate_pool(self, population: list[Chromosome]):
        mutated_pool = copy.deepcopy(population)
        for chromosome in mutated_pool:
            if random.random() <= self.prob_mut_chromosome:
                chromosome.mutate(self.input,self.prob_mut_gene)      
        return mutated_pool

    def carried_pool(self, population: list[Chromosome]):
        carry_size = int(self.prob_carry * self.population_size)
        carried_pool = sorted(population, key=lambda chromosome: chromosome.fitness, reverse=True)
        return copy.deepcopy(carried_pool[:carry_size])

    def evaluate(self, population: list[Chromosome]):
        for chromosome in population:
            if int(chromosome.fitness) == 1:
                return chromosome
        return None

    def find_ex(self):
        population = self.generate_population()
        cur_num_generation = 0
        while True:
            if cur_num_generation == self.max_generation:
                return population
            cur_num_generation += 1
            
            for chromosome in population:
                chromosome.calc_fitness(self.input)
            result = self.evaluate(population)
            print(max(chromosome.fitness for chromosome in population))
            if result is not None:
                return population

            carried = self.carried_pool(population)
            mating = self.mating_pool(population)
            random.shuffle(population)
            new_generation = self.crossover(mating)
            muted_generation = self.mutate_pool(new_generation)
            population = muted_generation[:self.population_size - len(carried)]
            population.extend(carried)


# input_data = Input(
#     ex_len=8,
#     coefficients=[x for x in range(-10, 10)],
#     goal={1: 7, 2: 527, 3: 9723, 0: 9, -1: 11, -2: -677, -3: -12153}
# )
input_data = Input(
    ex_len=4,
    coefficients=[x for x in range(-6, 10)],
    goal={0: 1, 1: 0, 2: -5, -1:-8}
)

curve_fitting = Genetic(
    input=input_data,
    prob_xover=0.4,
    prob_mut_chromosome=0.6,
    prob_mut_gene=0.7,
    prob_carry=0.09,
    population_size=100,
    max_generation=5000
)

result = curve_fitting.find_ex()

best = sorted(result, key=lambda x: -x.fitness)[0]
coefficients = best.genes
print(coefficients, best.fitness)
x = list(curve_fitting.input.goal.keys())
y = list(curve_fitting.input.goal.values())

x_curve = np.linspace(min(x), max(x), 100)

coefficients.reverse()
y_curve = np.polyval(coefficients, x_curve)
plt.plot(x, y, 'ro', label='Points')
plt.plot(x_curve, y_curve, label='Polynomial Curve')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of Points and Polynomial Curve')

plt.legend()

plt.show()