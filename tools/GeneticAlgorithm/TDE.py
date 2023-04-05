
import random
import torch
import copy
import numpy as np
from VGG16_Model import vgg


population_size = 50
generations = 100
F = 0.5
CR = 0.6
xmin = -0.1
xmax = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vgg().to(device)
model.load_state_dict(torch.load(r'C:\Users\LC\Desktop\AGSM-DE\AGSM-DE-PythonCode\vgg16_params.pkl', map_location=torch.device('cuda')))



# pop size (50, 3 , 608, 608)
def init_population(dim):
    population = np.zeros((population_size, dim))
    for i in range(population_size):
        for j in range(dim):
           rand_value = random.random()
           population[i,j] = xmin + rand_value * (xmax-xmin)
    return population



def calculate_fitness(taget_image, population, first_labels, dim):
    taget_image = taget_image.cpu().detach().numpy()
    fitness = []
    function_value=np.zeros(population_size)
    model.eval()
    for b in range(population_size):
       attack_image = taget_image + population[b, :, :, :]
       attack_image = torch.from_numpy(attack_image)
       attack_image = attack_image.to(device)
       outputs = model(attack_image.float())
       outputs = outputs.cpu().detach().numpy()
       d = outputs[0, first_labels]
       c = np.min(outputs)
       outputs.itemset(first_labels, c)
       g = np.max(outputs)
       function_value[b] = d-g
       fitness.append(function_value[b])

    return fitness


def mutation(population, dim):

    Mpopulation = np.zeros((population_size, dim))
    for i in range(population_size):
        r1 = r2 = r3 = 0
        while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
            r1 = random.randint(0, population_size - 1)
            r2 = random.randint(0, population_size - 1)
            r3 = random.randint(0, population_size - 1)
        Mpopulation[i] = population[r1] + F * (population[r2] - population[r3])

        for j in range(dim):
            if xmin <= Mpopulation[i, j] <= xmax:
                Mpopulation[i, j] = Mpopulation[i, j]
            else:
                Mpopulation[i,j] = xmin + random.random() * (xmax - xmin)
    return Mpopulation

def crossover(Mpopulation, population, dim):
  Cpopulation = np.zeros((population_size,dim))
  for i in range(population_size):
     for j in range(dim):
        rand_float = random.random()
        if rand_float <= CR:
             Cpopulation[i, j] = Mpopulation[i, j]
        else:
             Cpopulation[i, j] = population[i, j]
  return Cpopulation

def selection(taget_image, Cpopulation, population,first_labels, dim, pfitness):
    Cfitness = calculate_fitness(taget_image, Cpopulation,first_labels, dim)
    for i in range(population_size):
        if Cfitness[i] < pfitness[i]:
            population[i] = Cpopulation[i]
            pfitness[i] = Cfitness[i]
        else:
            population[i] = population[i]
            pfitness[i] = pfitness[i]
    return population, pfitness

def FDE(clean_image, org_labels):

    population = init_population(dim)
    fitness = calculate_fitness(clean_image, population, org_labels, dim)
    Best_indi_index = np.argmin(fitness)
    Best_indi = population[Best_indi_index, :]

    for step in range(generations):
        if min(fitness) < 0:
           break
        Mpopulation = mutation(population, dim)
        Cpopulation = crossover(Mpopulation, population, dim)
        population, fitness = selection(clean_image, Cpopulation, population, org_labels, dim, fitness)
        Best_indi_index = np.argmin(fitness)
        Best_indi = population[Best_indi_index, :]

    clean_image = clean_image.cpu().detach().numpy()
    final_image = clean_image + Best_indi
    final_image = torch.from_numpy(final_image)
    final_image = final_image.float()
    final_image[0, :, :, :] = torch.clamp(final_image[0, :, :, :], 0, 1)

    return final_image
