package models;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import utility.FuncUtils;

/**
 * conceptLDA: A Java package for the LDA and concept topic models
 * 
 * Implementation of the concept topic model, using
 * collapsed Gibbs sampling, as described in:
 * 
 * C. Chemudugunta, A. Holloway, P. Smyth, and M. Steyvers.
 * "Modeling documents by combining semantic concepts with unsupervised statistical learning,"
 * Proceedings of the International Semantic Web Conference (ISWC-08), 2008
 * 
 * @author: Sultan S. Alqathani
 */
public class GibbsSamplingConceptLDA {
	public double alpha; // Hyper-parameter alpha correspond to Dirichlet prior Theta 
	public double beta_phi; // Hyper-parameter beta correspond to Dirichlet prior Phi
	
	public double beta_psi; // Hyper-parameter beta correspond to Dirichlet prior Psi
	public int numTopics; // Number of topics

	public int numConcepts; // Number of concepts

	public int numIterations; // Number of Gibbs sampling iterations
	public int topWords; // Number of most probable words for each topic

	public double alphaSum; // alpha * numTopics
	public double betaSum_phi; // beta * vocabularySize

	public double betaSum_psi; // beta * conceptsWordsSize

	public List<List<Integer>> corpus; // Word ID-based corpus
	public List<List<Integer>> concepts; // Word ID-based concepts list
	
	public List<List<Integer>> topicAssignments; // Topics assignments for words
												// in the corpus
	
	public List<List<Integer>> conceptAssignments; // Concepts assignments for words
												   // in the corpus
	
	public int numDocuments; // Number of documents in the corpus
	public int numWordsInCorpus; // Number of words in the corpus
	public int numWordsInConcepts; // number of words in the concepts lists

	public HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID
														// given a word
	public HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word
														// given an ID
	
	public HashMap<String, Integer> concetpWord2IdVocabulary; // Word list to get ID
															// given a concept's word
	public HashMap<Integer, String> id2ConceptWordVocabulary; // World list get concept's word
															  // given an ID


	public int vocabularySize; // The number of word types in the corpus
	
	public int conceptsWordsSize; // The number of word types in the concepts lists

	// numDocuments * (numbConcepts + numTopics) matrix
	// Given a document: number of its words assigned to each topic or concept
	public int[][] docConceptTopicCount;
	// Number of words in every document
	public int[] sumDocConceptTopicCount;
	
	// numTopics * vocabularySize matrix
	// Given a topic: number of times a word type assigned to the topic
	public int[][] topicWordCount;
	// Total number of words assigned to a topic
	public int[] sumTopicWordCount;
	
	// numbConcepts * vocaularySize matrix
	// Given a concept: number of times a word type assigned to the concept
	public int[][] conceptWordCount;
	// Total number of words assigned to a concept
	public int[] sumConceptWordCount;

	// Double array used to sample a topic or concept
	public double[] multiPros;
		
	// Double array used to sample a topic 
	public double[] multiProsTopic;
	
	// Double array used to sample a concept
	public double[] multiProsConcept;

	// Path to the directory containing the corpus
	public String folderPath;
	// Path to the topic modeling corpus
	public String corpusPath;
	// Path to the concepts lists
	public String conceptsPath;

	public String expName = "LDAmodel";
	public String orgExpName = "LDAmodel";
	public String tAssignsFilePath = "";
	public int savestep = 0;


	public GibbsSamplingConceptLDA(String pathToCorpus, String pathToConcepts, int inNumTopics, int inNumConcepts,
		double inAlpha, double inBeta, int inNumIterations, int inTopWords, String inExpName)
		throws Exception
	{

		alpha = inAlpha;
		beta_phi = inBeta;
		beta_psi = inBeta;
		numTopics = inNumTopics;
		numConcepts = inNumConcepts;
		numIterations = inNumIterations;
		topWords = inTopWords;
	//	savestep = inSaveStep;
		expName = inExpName;
		orgExpName = expName;
		corpusPath = pathToCorpus;
		conceptsPath = pathToConcepts;
		folderPath = pathToCorpus.substring(
			0,
			Math.max(pathToCorpus.lastIndexOf("/"),
				pathToCorpus.lastIndexOf("\\")) + 1);

		System.out.println("Reading topic modeling corpus: " + pathToCorpus);
		System.out.println("Reading concept lists: "+ pathToConcepts);

		word2IdVocabulary = new HashMap<String, Integer>();
		id2WordVocabulary = new HashMap<Integer, String>();
		
		concetpWord2IdVocabulary = new HashMap<String, Integer>();
		id2ConceptWordVocabulary = new HashMap<Integer, String>();
		
		corpus = new ArrayList<List<Integer>>();
		concepts = new ArrayList<List<Integer>>();
		
		numDocuments = 0;
		numWordsInCorpus = 0;
		numWordsInConcepts = 0;

		BufferedReader br_ = null;
		try{
			int indexWord = -1;
			br_ = new BufferedReader(new FileReader(pathToConcepts));
			for (String con; (con = br_.readLine()) != null;){
				if (con.trim().length() == 0)
					continue;
				String[] words = con.trim().split("\\s+");
				List<Integer> concept = new ArrayList<Integer>();
				for (String word : words){
					if(concetpWord2IdVocabulary.containsKey(word)){
						concept.add(concetpWord2IdVocabulary.get(word));
					}else{
						indexWord +=1;
						concetpWord2IdVocabulary.put(word, indexWord);
						id2ConceptWordVocabulary.put(indexWord, word);
						concept.add(indexWord);
					}
				}
				numWordsInConcepts += words.length;
				concepts.add(concept);
			}
		}catch(Exception e){
			e.printStackTrace();
		}
		BufferedReader br = null;
		try {
			int indexWord = -1;
			br = new BufferedReader(new FileReader(pathToCorpus));
			for (String doc; (doc = br.readLine()) != null;) {

				if (doc.trim().length() == 0)
					continue;

				String[] words = doc.trim().split("\\s+");
				List<Integer> document = new ArrayList<Integer>();

				for (String word : words) {
					if (word2IdVocabulary.containsKey(word)) {
						document.add(word2IdVocabulary.get(word));
					}
					else {
						indexWord += 1;
						word2IdVocabulary.put(word, indexWord);
						id2WordVocabulary.put(indexWord, word);
						document.add(indexWord);
					}
				}

				numDocuments++;
				numWordsInCorpus += document.size();
				corpus.add(document);
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}

		vocabularySize = word2IdVocabulary.size(); // vocabularySize = indexWord
		conceptsWordsSize = concetpWord2IdVocabulary.size(); // conceptswordsSize
		docConceptTopicCount = new int[numDocuments][numTopics+numConcepts];
		topicWordCount = new int[numTopics][vocabularySize];
		conceptWordCount = new int[numConcepts][vocabularySize];
		sumDocConceptTopicCount = new int[numDocuments];
		sumTopicWordCount = new int[numTopics];
		sumConceptWordCount = new int[numConcepts];
		
		multiProsTopic = new double[numTopics];
		for (int i = 0; i < numTopics; i++) {
			multiProsTopic[i] = 1.0 / (numTopics);
		}
		
		multiProsConcept = new double[numConcepts];
		for (int i = 0; i < numConcepts; i++) {
			multiProsConcept[i] = 1.0 / (numConcepts);
		}
		
		multiPros = new double[numTopics+numConcepts];
		for (int i = 0; i < numTopics+numConcepts; i++) {
			multiPros[i] = 1.0 / (numTopics+numConcepts);
		}

		alphaSum = numTopics * alpha;
		betaSum_phi = vocabularySize * beta_phi;
		betaSum_psi = conceptsWordsSize * beta_psi;

		System.out.println("Corpus size: " + numDocuments + " docs, "
			+ numWordsInCorpus + " words");
		System.out.println("Vocabuary size: " + vocabularySize);
		System.out.println("Concepts word size: " + conceptsWordsSize);
		System.out.println("Number of topics: " + numTopics);
		System.out.println("Number of concepts: " + numConcepts);
		System.out.println("alpha: " + alpha);
		System.out.println("beta_phi: " + beta_phi);
		System.out.println("beta_psi: " + beta_psi);
		System.out.println("Number of sampling iterations: " + numIterations);
		System.out.println("Number of top topical words: " + topWords);

/*		tAssignsFilePath = pathToTAfile;
		if (tAssignsFilePath.length() > 0)
			initialize(tAssignsFilePath);
		else*/
			initialize();
	}

	/**
	 * Randomly initialize topic and concept assignments
	 */
	public void initialize()
		throws IOException
	{
		System.out.println("Randomly initializing topic assignments ...");

		topicAssignments = new ArrayList<List<Integer>>();
		conceptAssignments = new ArrayList<List<Integer>>();
		
		for (int i = 0; i < numDocuments; i++) {
			List<Integer> topics = new ArrayList<Integer>();
			List<Integer> concepts = new ArrayList<Integer>();
			int docSize = corpus.get(i).size();
			for (int j = 0; j < docSize; j++) {
				int topic = FuncUtils.nextDiscrete(multiProsTopic); // Sample a topic
				int concept = FuncUtils.nextDiscrete(multiProsConcept); // Sample a concept
//				int conceptTopic = FuncUtils.nextDiscrete(multiPros); // Sample a concept
				// Increase counts
				docConceptTopicCount[i][concept+topic] += 1;
				topicWordCount[topic][corpus.get(i).get(j)] += 1;
				conceptWordCount[concept][corpus.get(i).get(j)] +=1;
				sumDocConceptTopicCount[i] += 1;
				sumTopicWordCount[topic] += 1;
				sumConceptWordCount[concept] += 1;

				topics.add(topic);
				concepts.add(concept);
			}
			topicAssignments.add(topics);
			conceptAssignments.add(concepts);
		}
	}

//	/**
//	 * Initialize topic assignments from a given file
//	 */
//	public void initialize(String pathToTopicAssignmentFile)
//	{
//		System.out.println("Reading topic-assignment file: "
//			+ pathToTopicAssignmentFile);
//
//		topicAssignments = new ArrayList<List<Integer>>();
//
//		BufferedReader br = null;
//		try {
//			br = new BufferedReader(new FileReader(pathToTopicAssignmentFile));
//			int docID = 0;
//			int numWords = 0;
//			for (String line; (line = br.readLine()) != null;) {
//				String[] strTopics = line.trim().split("\\s+");
//				List<Integer> topics = new ArrayList<Integer>();
//				for (int j = 0; j < strTopics.length; j++) {
//					int topic = new Integer(strTopics[j]);
//					// Increase counts
//					docTopicCount[docID][topic] += 1;
//					topicWordCount[topic][corpus.get(docID).get(j)] += 1;
//					sumDocTopicCount[docID] += 1;
//					sumTopicWordCount[topic] += 1;
//
//					topics.add(topic);
//					numWords++;
//				}
//				topicAssignments.add(topics);
//				docID++;
//			}
//
//			if ((docID != numDocuments) || (numWords != numWordsInCorpus)) {
//				System.out
//					.println("The topic modeling corpus and topic assignment file are not consistent!!!");
//				throw new Exception();
//			}
//		}
//		catch (Exception e) {
//			e.printStackTrace();
//		}
//	}

	public void inference()
		throws IOException
	{
		writeParameters();
		writeDictionary();

		System.out.println("Running Gibbs sampling inference: ");

		for (int iter = 1; iter <= numIterations; iter++) {

			System.out.println("\tSampling iteration: " + (iter));
			// System.out.println("\t\tPerplexity: " + computePerplexity());

			sampleInSingleIteration();

			if ((savestep > 0) && (iter % savestep == 0)
				&& (iter < numIterations)) {
				System.out.println("\t\tSaving the output from the " + iter
					+ "^{th} sample");
				expName = orgExpName + "-" + iter;
				write();
			}
		}
		expName = orgExpName;

		System.out.println("Writing output from the last sample ...");
		write();

		System.out.println("Sampling completed!");

	}

	public void sampleInSingleIteration()
	{
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			int docSize = corpus.get(dIndex).size();
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				// Get current word and its topic
				int topic = topicAssignments.get(dIndex).get(wIndex);
				int word = corpus.get(dIndex).get(wIndex);
				
				int concept = conceptAssignments.get(dIndex).get(wIndex);

				// Decrease counts
				docConceptTopicCount[dIndex][topic+concept] -= 1;
				// docTopicSum[dIndex] -= 1;
				topicWordCount[topic][word] -= 1;
				conceptWordCount[concept][word] -= 1;
				sumTopicWordCount[topic] -= 1;
				sumConceptWordCount[concept] -= 1;
				
				if(concetpWord2IdVocabulary.containsKey(id2WordVocabulary.get(word))){
					// Sample a concept
					for(int cIndex = 0; cIndex < numConcepts; cIndex++){
						multiProsConcept[cIndex] = (docConceptTopicCount[dIndex][cIndex] + alpha)
								* ((conceptWordCount[cIndex][word] + beta_psi) / (sumConceptWordCount[cIndex] + betaSum_psi));
					}
					concept = FuncUtils.nextDiscrete(multiProsConcept);
				}else{
					// Sample a topic
					for (int tIndex = 0; tIndex < numTopics; tIndex++) {
						multiProsTopic[tIndex] = (docConceptTopicCount[dIndex][tIndex] + alpha)
							* ((topicWordCount[tIndex][word] + beta_phi) / (sumTopicWordCount[tIndex] + betaSum_phi));
						// multiPros[tIndex] = ((docTopicCount[dIndex][tIndex] +
						// alpha) /
						// (docTopicSum[dIndex] + alphaSum))
						// * ((topicWordCount[tIndex][word] + beta) /
						// (topicWordSum[tIndex] + betaSum));
					}
					topic = FuncUtils.nextDiscrete(multiProsTopic);
				}

				// Increase counts
				docConceptTopicCount[dIndex][topic+concept] += 1;
				// docTopicSum[dIndex] -= 1;
				topicWordCount[topic][word] += 1;
				conceptWordCount[concept][word] += 1;
				sumTopicWordCount[topic] += 1;
				sumConceptWordCount[concept] += 1;

				// Update topic assignments
				topicAssignments.get(dIndex).set(wIndex, topic);
				// Update concept assignments
				conceptAssignments.get(dIndex).set(wIndex, concept);
			}
		}
	}

	public void writeParameters()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".paras"));
		writer.write("-model" + "\t" + "LDA");
		writer.write("\n-corpus" + "\t" + corpusPath);
		writer.write("\n-concepts" + "\t" + conceptsPath);
		writer.write("\n-ntopics" + "\t" + numTopics);
		writer.write("\n-nconcepts" + "\t" + numConcepts);
		writer.write("\n-alpha" + "\t" + alpha);
		writer.write("\n-beta_phi" + "\t" + beta_phi);
		writer.write("\n-beta_psi" + "\t" + beta_psi);
		writer.write("\n-niters" + "\t" + numIterations);
		writer.write("\n-twords" + "\t" + topWords);
		writer.write("\n-name" + "\t" + expName);
		if (tAssignsFilePath.length() > 0)
			writer.write("\n-initFile" + "\t" + tAssignsFilePath);
		if (savestep > 0)
			writer.write("\n-sstep" + "\t" + savestep);

		writer.close();
	}

	public void writeDictionary()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".vocabulary"));
		for (int id = 0; id < vocabularySize; id++)
			writer.write(id2WordVocabulary.get(id) + " " + id + "\n");
		writer.close();
	}

	public void writeIDbasedCorpus()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".IDcorpus"));
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			int docSize = corpus.get(dIndex).size();
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				writer.write(corpus.get(dIndex).get(wIndex) + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void writeTopicAssignments()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".topicAssignments"));
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			int docSize = corpus.get(dIndex).size();
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				writer.write(topicAssignments.get(dIndex).get(wIndex) + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}
	
	public void writeConceptAssignments()
			throws IOException
		{
			BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
				+ expName + ".conceptAssignments"));
			for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
				int docSize = corpus.get(dIndex).size();
				for (int wIndex = 0; wIndex < docSize; wIndex++) {
					writer.write(conceptAssignments.get(dIndex).get(wIndex) + " ");
				}
				writer.write("\n");
			}
			writer.close();
		}

	public void writeTopTopicalWords()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".topWords"));

		for (int tIndex = 0; tIndex < numTopics; tIndex++) {
			writer.write("Topic" + new Integer(tIndex) + ":");

			Map<Integer, Integer> wordCount = new TreeMap<Integer, Integer>();
			for (int wIndex = 0; wIndex < vocabularySize; wIndex++) {
				wordCount.put(wIndex, topicWordCount[tIndex][wIndex]);
			}
			wordCount = FuncUtils.sortByValueDescending(wordCount);

			Set<Integer> mostLikelyWords = wordCount.keySet();
			int count = 0;
			for (Integer index : mostLikelyWords) {
				if (count < topWords) {
					double pro = (topicWordCount[tIndex][index] + beta_phi)
						/ (sumTopicWordCount[tIndex] + betaSum_phi);
					pro = Math.round(pro * 1000000.0) / 1000000.0;
					writer.write(" " + id2WordVocabulary.get(index) + "(" + pro
						+ ")");
					count += 1;
				}
				else {
					writer.write("\n\n");
					break;
				}
			}
		}
		writer.close();
	}

	public void writeTopConceptWords() throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName + ".topConceptWords"));

		for (int cIndex = 0; cIndex < numConcepts; cIndex++) {
			writer.write("Concept" + new Integer(cIndex) + ":");

			Map<Integer, Integer> wordCount = new TreeMap<Integer, Integer>();
			for (int wIndex = 0; wIndex < conceptsWordsSize; wIndex++) {
				wordCount.put(wIndex, conceptWordCount[cIndex][wIndex]);
			}
			wordCount = FuncUtils.sortByValueDescending(wordCount);

			Set<Integer> mostLikelyWords = wordCount.keySet();
			int count = 0;
			for (Integer index : mostLikelyWords) {
				if (count < topWords) {
					double pro = (conceptWordCount[cIndex][index] + beta_psi)
							/ (sumConceptWordCount[cIndex] + betaSum_psi);
					pro = Math.round(pro * 1000000.0) / 1000000.0;
					writer.write(" " + id2ConceptWordVocabulary.get(index) + "(" + pro + ")");
					count += 1;
				} else {
					writer.write("\n\n");
					break;
				}
			}
		}
		writer.close();
	}
	public void writeTopicWordPros()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".phi"));
		for (int i = 0; i < numTopics; i++) {
			for (int j = 0; j < vocabularySize; j++) {
				double pro = (topicWordCount[i][j] + beta_phi)
					/ (sumTopicWordCount[i] + betaSum_phi);
				writer.write(pro + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void writeConceptWordPros()
			throws IOException
		{
			BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
				+ expName + ".psi"));
			for (int i = 0; i < numConcepts; i++) {
				for (int j = 0; j < conceptsWordsSize; j++) {
					double pro = (conceptWordCount[i][j] + beta_psi)
						/ (sumConceptWordCount[i] + betaSum_psi);
					writer.write(pro + " ");
				}
				writer.write("\n");
			}
			writer.close();
		}
	public void writeTopicWordCount()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".WTcount"));
		for (int i = 0; i < numTopics; i++) {
			for (int j = 0; j < vocabularySize; j++) {
				writer.write(topicWordCount[i][j] + " ");
			}
			writer.write("\n");
		}
		writer.close();

	}
	
	public void writeConceptWordCount()
			throws IOException
		{
			BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
				+ expName + ".WCcount"));
			for (int i = 0; i < numConcepts; i++) {
				for (int j = 0; j < conceptsWordsSize; j++) {
					writer.write(conceptWordCount[i][j] + " ");
				}
				writer.write("\n");
			}
			writer.close();

		}

	public void writeDocTopicPros()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".theta"));
		for (int i = 0; i < numDocuments; i++) {
			for (int j = 0; j < numTopics+numConcepts; j++) {
				double pro = (docConceptTopicCount[i][j] + alpha)
					/ (sumDocConceptTopicCount[i] + alphaSum);
				writer.write(pro + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void writeDocTopicCount()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".DTcount"));
		for (int i = 0; i < numDocuments; i++) {
			for (int j = 0; j < numTopics; j++) {
				writer.write(docConceptTopicCount[i][j] + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void write()
		throws IOException
	{
		writeTopTopicalWords();
		writeDocTopicPros();
		writeTopicAssignments();
		writeTopicWordPros();
		
		writeTopConceptWords();
		writeConceptAssignments();
		writeConceptWordPros();
	}

	public static void main(String args[])
		throws Exception
	{
		GibbsSamplingConceptLDA clda = new GibbsSamplingConceptLDA("test/corpus.txt", "test/concepts.txt", 7,2, 0.1, 0.01, 2000, 20, "testLDA");
		clda.inference();
	}
}
