import numpy as np
import random
from PIL import Image, ImageOps
import math


# initializes the population
def initialize(width, height, pop_size, num_circle):
    initial_pop = []
    for _ in range(pop_size):
        circles = []
        for i in range(num_circle):
            radius = random.randint(0, width)
            x = random.randint(0, width)
            y = random.randint(0, height)
            circle = [x, y, radius, random.randint(0, 255)]
            circles.append(circle)
        initial_pop.append(circles)
    return initial_pop


# measures deviation from target image on pixel by pixel basis
def fitness(width, height, guess, target, num_circle):

    score = 0
    for y in range(height):
        for x in range(width):
            colour = 0
            num_overlap = 0
            pixel_fitness = 0
            for i in range(num_circle):
                # detects if a x,y can solve one or more equations of a circle
                if (x - guess[i][0]) ** 2 + (y - guess[i][1]) ** 2 <= guess[i][2] ** 2:
                    num_overlap += 1
                    colour += guess[i][3]
            if num_overlap >= 1:
                # calculates average colour on a pixel when circles overlap
                colour = colour // num_overlap
                pixel_fitness = math.sqrt((target[y][x] - colour) ** 2)
                # lower score is better
            score += pixel_fitness
    return score


# breeds two genotypes by switching one or more circle with each other
def breed(parent1, parent2, num_circle):

    for i in range(num_circle):
        if random.randint(0, 1) == 1:
            holder = parent1[i]
            parent1[i] = parent2[i]
            parent2[i] = holder
    return parent1, parent2


# mutates a genotype by a certain mutation rate
def mutate(child, rate, num_circle):
    # mutate by changing colour of circle but not position
    if random.randint(0, rate - 1) == 0:
        for _ in range(random.randint(0, (num_circle - 1) // 4)):
            index = random.randint(0, num_circle - 1)
            child[index][3] = random.randint(0, 255)
    return child


# sorts fitness from low to high
def sort_fitness(target, population, width, height, num_circle):
    population_fitness = []
    for guess in population:
        fit = fitness(width, height, guess, target, num_circle)
        population_fitness.append([guess, fit])

    x = sorted(population_fitness, key=lambda x: int(x[1]), reverse=False)
    return x


# makes room for next generation by more likely killing worst part of population
def kill(sorted_population, new_pop_size):
    # kills the worst
    best = round(len(sorted_population) * 0.2)
    new_pop = sorted_population[:best]
    sorted_population = sorted_population[best:]
    while len(new_pop) < new_pop_size:
        index = random.randint(0, len(sorted_population) - 1)
        new_pop.append(sorted_population[index])
    return new_pop


# regenerates the population back to the pop size
def regeneration(new_population, pop_size, mutation_rate, num_circle):

    next_gen = []
    while len(new_population) + len(next_gen) < pop_size:
        mate1 = ""
        mate2 = ""
        while mate1 == mate2:
            mate1 = random.randint(0, len(new_population) - 1)
            mate2 = random.randint(0, len(new_population) - 1)

        child1, child2 = breed(new_population[mate1][0], new_population[mate2][0], num_circle)
        next_gen.append(child1)
        next_gen.append(child2)

    for genome in new_population:
        next_gen.append(genome[0])

    for index in range(len(next_gen)):
        next_gen[index] = mutate(next_gen[index], mutation_rate, num_circle)
    return next_gen


# calculates mean from a list
def mean_2_d(list):

    sum = 0
    for i in range(len(list)):
        sum += list[i][1]

    return sum / len(list)


# will add a random circle to every genotype in pop
def add_circle(pop, pop_size, width, height):

    for i in range(pop_size):
        radius = random.randint(0, width)
        x = random.randint(0, width)
        y = random.randint(0, height)
        circle = [x, y, radius, random.randint(0, 255)]
        pop[i][0].append(circle)
    return pop


# converts a genotype to a jpeg
def convert_to_image(guess, num_circle, height, width, generation_num):

    # creates empty numpy array
    image = np.zeros((height, width), np.int8)

    for y in range(height):
        for x in range(width):
            colour = 0
            num_overlap = 0
            for i in range(num_circle):
                # calculates average colour
                if (x - guess[i][0]) ** 2 + (y - guess[i][1]) ** 2 <= guess[i][2] ** 2:
                    num_overlap += 1
                    colour += guess[i][3]
            # adds pixel value to array
            image[y][x] = colour // num_circle
    # converts numpy array to image object
    im = Image.fromarray(image, 'L')
    # saves image to file path as jpeg
    im.save("C:/filepath/generation" + str(generation_num) + ".jpeg")


def genetic_guess(target, pop_size, height, width, num_circle, max_circle, mutation_rate, feedback, reporting_time):
    # initializes population
    initial = initialize(width, height, pop_size, num_circle)
    # sorts initial gen by fitness
    sorted_gen = sort_fitness(target, initial, width, height, num_circle)

    # prev_mean and prev_high keep track of the mean and high of the last 5 generations fitness
    prev_mean = [0, 0, 0, 0, 0]
    prev_high = [0, 0, 0, 0, 0]
    # calls mean_2_d to find mean fitness
    mean = mean_2_d(sorted_gen)
    high = sorted_gen[0][1]
    generation_num = 1
    keepGoing = True

    while keepGoing:
        # checks to see if current mean and high are less than or equal past 5 generations
        if all(mean <= i for i in prev_mean) and all(high <= i for i in prev_high):
            # will add a circle to increase detail in solution
            if num_circle < max_circle:
                print("added circle")
                num_circle += 1
                print(num_circle)
                add_circle(sorted_gen, pop_size, width, height)
            else:
                # if fitness hasn't changed in last 5 generations and at max circles must be at best solution
                keepGoing = False

        new_pop = kill(sorted_gen, pop_size * .5)

        next_gen = regeneration(new_pop, pop_size, mutation_rate, num_circle)

        sorted_gen = sort_fitness(target, next_gen, width, height, num_circle)

        # converts best genotype of population to image every x generations
        if feedback and generation_num % reporting_time == 0:
            convert_to_image(sorted_gen[0][0], num_circle, height, width, generation_num)

        # updates prev mean and high lists
        prev_mean.pop(0)
        prev_mean.append(mean)
        prev_high.pop(0)
        prev_high.append(high)

        mean = mean_2_d(sorted_gen)
        high = sorted_gen[0][1]

        print("generation " + str(generation_num))
        generation_num += 1


image = Image.open("filepath")  # file path to image want to recreate
image = ImageOps.grayscale(image)  # changing image to grayscale to help with compute time
a = np.asarray(image)  # turning image into 2-d numpy array of pixels
width, height = image.size
genetic_guess(a, 30, height, width, 5, 200, 2, True, 10)
# using array a, pop_size of 30, 5 initial circles, 200 max circles, mutation rate of 2, saves best image every 10
# generations
