import random
import math
from optparse import OptionParser
from operator import itemgetter

def parseArguments():
	parser = OptionParser()
	parser.add_option('-f', '--file', dest='filename',
		help='input file name', type="string")
	parser.add_option('-n', '--ncount', dest='ncount',
		help='nodes in middle layer', default=2, type="int")
	parser.add_option('-t', '--threshold', dest='threshold',
		help='neural network treshold', default=5.0, type="float")
	parser.add_option('-s', '--size', dest='populationSize',
		help='size of initial population', default=20, type="int")
	parser.add_option('-i', '--iter', dest='iterations',
		help='number of iterations', default=100, type="int")
	parser.add_option('-e', '--errthresh', dest='errthresh',
		help='error threshold', default=0.0, type="float")
	parser.add_option('-p', '--mutation', dest='mutProb',
		help='probability of weight mutation', default=0.05, type="float")
	parser.add_option('-K', dest='K', default=1, type="float")
	parser.add_option('-T', '--testfile', dest='testfile',
		help='name of file for testing network', type="string")
	return parser.parse_args()

def parseFileContents(filename):
	f = open(filename)
	contents = map(lambda l: tuple(l.strip().split()), f.readlines()[1:])
	tuples = map(lambda (x, fx): (float(x), float(fx)), contents)
	f.close()
	return tuples

def generateRange(count, low, high):
	return [random.uniform(low, high) for _ in xrange(count)]

def stddev(results):
	n = len(results)
	if n == 0:
		raise StandardError('Empty Population')
	sqrDiffs = []
	for x, fx in results:
		sqrDiffs.append((x - fx) ** 2)
	return sum(sqrDiffs) / n

def sigmoidal(x, a=1):
	try:
		return 1 / (1 + math.exp(- a * x))
	except OverflowError:
		return 1

def identity(x):
	return x

class Network:

	ID = 0
	LOW = -10
	HIGH = 10

	def __init__(self, middle, out):
		self._middleLayer = middle
		self._outNode = out
		self._n = len(middle) / 2
		self._id = Network.ID
		Network.ID += 1

	@classmethod
	def createRandom(cls, n, threshold):
		middle = [Network.Node.createRandom(threshold, 1, sigmoidal) for _ in xrange(n)]
		outNode = Network.Node.createRandom(threshold, n, identity)
		return cls(middle, outNode)

	@classmethod
	def createFromChromosome(cls, chromosome):
		n = len(chromosome)
		size = (n - 1) / 3
		middle = []
		for i in xrange(size):
			w1 = chromosome[2*i]
			w2 = chromosome[2*i+1]
			middle.append(Network.Node([w1, w2], sigmoidal))
		out = Network.Node(chromosome[2*size:], identity)
		return cls(middle, out)

	def evaluate(self, x):
		outXs = [node.out([x]) for node in self._middleLayer]
		return self._outNode.out(outXs)

	def evalNetwork(self, evalSet):
		results = []
		for x, fx in evalSet:
			val = self.evaluate(x)
			results.append((val, fx))
		return stddev(results)

	def chromosome(self):
		chrom = []
		for node in self._middleLayer:
			chrom += node.weights()
		chrom += self._outNode.weights()
		return chrom

	def __repr__(self):
		return 'Net(' + str(self._id) + ')'

	class Node:

		def __init__(self, weights, transFun):
			self._weights = weights
			self._transFun = transFun

		@classmethod
		def createRandom(cls, threshold, inCount, transFun):
			weights = [-threshold] + generateRange(inCount, Network.LOW, Network.HIGH)
			return cls(weights, transFun)

		def net(self, xs):
			xs = [1] + xs
			return sum(map(lambda (x, w): w * x, zip(xs, self._weights)))

		def out(self, xs):
			return self._transFun(self.net(xs))

		def weights(self):
			return self._weights

		def __repr__(self):
			return 'Node(' + str(self._weights) + ')'

class Genetics:

	def __init__(self, initial, maxIter, minError, mutProb, K):
		self._current = initial
		self._iter = 0
		self._maxIter = maxIter
		self._minError = minError
		self._mutProb = mutProb
		self._K = K

	def train(self, trainSet):
		while True:
			evaluated = self.evaluate(trainSet)
			elite = evaluated[0]
			print 'Iter:', self._iter, 'best:', elite[0]
			self._iter += 1
			if elite[0] < self._minError:
				print 'Gotovo: error threshold'
				return elite[1]
			if self._iter >= self._maxIter:
				print 'Gotovo: max iter'
				return elite[1]
			self._current = self.selection(evaluated)

	def evaluate(self, evalSet):
		evalResults = []
		for unit in self._current:
			evalResults.append((unit.evalNetwork(evalSet), unit))
		return sorted(evalResults, key=itemgetter(0))

	def selection(self, evaluated):
		evalList = zip(*evaluated)
		evalVals = evalList[0]
		evalUnits = evalList[1]
		fitness = self.Fitness(evalVals)
		roulette = self.Roulette(evalUnits, fitness)
		
		elite = evaluated[0][1]
		selected = [elite]
		n = len(self._current) - 1
		for i in xrange(n):
			parent1 = roulette.getRandom().chromosome()
			parent2 = roulette.getRandom().chromosome()
			crossed = self.cross(parent1, parent2)
			mutated = self.mutate(crossed)
			selected.append(Network.createFromChromosome(mutated))
		return selected

	def cross(self, chromosome1, chromosome2):
		crossed = []
		for i in xrange(len(chromosome1)):
			mean = (chromosome1[i] + chromosome2[i]) / 2
			crossed.append(mean)
		return crossed

	def mutate(self, chromosome):
		mutated = []
		for g in chromosome:
			p = random.random()
			if p < self._mutProb:
				norm = random.gauss(0, self._K)
				mutated.append(g + norm)
			else:
				mutated.append(g)
		return mutated

	class Fitness:

		def __init__(self, evalVals):
			subSum = evalVals[0] + evalVals[len(evalVals)-1]
			subVals = [subSum - val for val in evalVals]
			divSum = sum(subVals)
			self._fit = map(lambda val: val / divSum, subVals)

		def fit(self, i):
			return self._fit[i]

	class Roulette:

		def __init__(self, evalUnits, fitness):
			self._barrel = []
			self._units = evalUnits
			lim = 0
			for i in xrange(len(evalUnits)):
				f = fitness.fit(i)
				self._barrel.append(lim + f)
				lim += f

		def get(self, p):
			for i in xrange(len(self._units)):
				if p < self._barrel[i]:
					return self._units[i]
			return self._units[len(self._units) - 1]

		def getRandom(self):
			p = random.random()
			return self.get(p)


def generateNetworks(size, nodeCount, threshold):
	return [Network.createRandom(nodeCount, threshold) for _ in xrange(size)]

if __name__ == '__main__':
	opt, _ = parseArguments()

	networks = generateNetworks(opt.populationSize, opt.ncount, opt.threshold)
	gen = Genetics(networks, opt.iterations, opt.errthresh, opt.mutProb, opt.K)

	if opt.filename is not None:
		xfx = parseFileContents(opt.filename)
		best = gen.train(xfx)

		if opt.testfile is not None:
			testxfx = parseFileContents(opt.testfile)
			print 'Greska na testnom skupu', best.evalNetwork(testxfx)

		while True:
			print 'Upisi broj za isprobat:'
			x = float(raw_input())
			print best.evaluate(x)