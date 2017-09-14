import java.io.FileInputStream;
import java.io.IOException;

import javax.xml.parsers.ParserConfigurationException;

import org.xml.sax.SAXException;


//Jason Natale CSC242 Project 3
//Uncertain Inference
//First perform rejection sampling, then likelihood weighting
public class MyBNApproxInferencer {
	public static void main(String[] args){
		//receive num samples, file representing BN, query var, and
		//evidence vars
		int numS = Integer.parseInt(args[0]);
		String filename = args[1];
		String queryvar = args[2];
		int len = args.length;
		int numEvid = (len-3)/2;
		String[] enames = new String[numEvid];
		String[] evals = new String[numEvid];
		//loop through evidence vars
		for(int i=0;i<numEvid;i++){
			enames[i] = args[i*2+ 3];
			evals[i] = args[i*2 + 4];
		}
		String end = filename.substring(filename.length()-3);
		boolean xml = false;
		if(end.equals("xml"))
			xml = true;
		//convert file to Bayesian network
		BayesianNetwork bae = new BayesianNetwork();
		if(xml){
			XMLBIFParser parser = new XMLBIFParser();
			try{
				bae = parser.readNetworkFromFile(filename);
			}
			//errors in case parser cannot read file
			catch(IOException e){error(1,e.getMessage());}
			catch(SAXException e){error(2,e.getMessage());}
			catch(ParserConfigurationException e){error(3,e.getMessage());}
		}
		else{
			//dealing with .bif
			try{
			FileInputStream fis = new FileInputStream(filename);
			//need to create parser from input stream (file)
			BIFLexer lex = new BIFLexer(fis);
			BIFParser parser = new BIFParser(lex);
			bae = parser.parseNetwork();
			}
			catch(IOException e){
				System.out.println("Error reading .bif");
			}
		}
		//get possible assignments of query var
		RandomVariable qv = bae.getVariableByName(queryvar);
		//make assignments
		Assignment assign = new Assignment();
		for(int a=0;a<numEvid;a++){
			assign.set(bae.getVariableByName(enames[a]), evals[a]);
		}
		System.out.println("Size of network: "+bae.getVariableList().size());
		MyBNApproxInferencer tester = new MyBNApproxInferencer();
		System.out.println("Rejection Sampling:");
		double t = System.currentTimeMillis();
		Distribution distrib = tester.rejectsamp(bae,qv,assign,numS);
		System.out.println("Time taken: "+(System.currentTimeMillis()-t)/1000);
		//output distribution
		System.out.println(distrib.toString());
		//now do likelihood weighting
		System.out.println("\n\nLikelihood Weighting:");
		t = System.currentTimeMillis();
		distrib = tester.likelihood(bae,qv,assign,numS);
		System.out.println("Time taken: "+(System.currentTimeMillis()-t)/1000);
		System.out.println(distrib.toString());
	}
	
	//creates class object
	public MyBNApproxInferencer(){}
	
	//deliver unique error message if BN can't be created
	public static void error(int a, String msg){
		System.out.println("error making BN "+msg);
	}
	
	//Calculate the distribution of a variable with rejection sampling
	public Distribution rejectsamp(BayesianNetwork bn, RandomVariable X, Assignment e,int numsamp){
		Distribution d = new Distribution(X);
		//counts number of samples valid for each setting of var X
		int[] counts = new int[X.domain.size()];
		//perform sampling iteratively
		for(int j=1;j<=numsamp;j++){
			Assignment samp = GeneratePriorSample(bn);
			if(consistent(e,samp)){
				//find what value it is and increment that counts[domainval]
				for(int i=0;i<X.domain.size();i++){
					if(samp.get(X).equals(X.domain.get(i))){
						counts[i]++;
					}
				}
			}
		}
		//put values into a distribution
		for(int i=0;i<X.domain.size();i++){
			d.put(X.domain.get(i), counts[i]);
		}
		//normalize d before returning
		d.normalize();
		return d;
	}
	
	//Calc distrib with likelihood weighting
	public Distribution likelihood(BayesianNetwork bn, RandomVariable X, Assignment e, int num){
		Distribution d = new Distribution(X);
		//set distrib to zero
		for(int i=0;i<X.domain.size();i++){
			d.put(X.domain.get(i), 0);
		}
		//iterate through samples
		for(int j=1;j<=num;j++){
			WeightedSample ws = WeightedSample(bn,e);
			//get value of Random Var from sample
			double curr = d.get(ws.ass.get(X));
			d.put(ws.ass.get(X), curr+ws.weight);
		}
		//normalize d before returning
		d.normalize();
		return d;
	}
	
	//returns the weight of the sample calculated
	public WeightedSample WeightedSample(BayesianNetwork bn, Assignment e){
		//loop through all vars, 
		//when coming across one which is set, mult the weight 
		//(prob of getting the value)
		WeightedSample ws = new WeightedSample();
		//initialize var values and weight
		ws.ass = new Assignment();
		ws.weight = 1;
		for(RandomVariable RV : bn.getVariableListTopologicallySorted()){
			//if assigned calc probability
			if(e.containsKey(RV)){
				//add to assignment the value stated, get condProb
				ws.ass.put(RV, e.get(RV));
				ws.weight *= bn.getProb(RV, ws.ass);
			}
			else{
				//get random sample and add to assignment
				Distribution d = new Distribution();
				for(int k=0;k<RV.domain.size();k++){
					Assignment a = new Assignment(ws.ass);
					a.set(RV, RV.domain.get(k));
					double res = bn.getProb(RV, a);
					d.put(RV.domain.get(k), res);
				}
				d.normalize();
				double val = Math.random();
				//now test to see which value of the domain has been chosen
				double tot = 0.0;
				for(int i=0;i<RV.domain.size();i++){
					tot += d.get(RV.domain.get(i));
					if(val<= tot){
						ws.ass.set(RV, RV.domain.get(i));
						break;
					}
				}
			}
		}
		return ws;
	}
	
	//generates a sample assignment of a bayesian network
	public Assignment GeneratePriorSample(BayesianNetwork bn){
		Assignment ret = new Assignment();
		//loop through all vars of network, assigning them
		for(RandomVariable RV : bn.getVariableListTopologicallySorted()){
			//get prob of each vars values
			Distribution d = new Distribution();
			for(int k=0;k<RV.domain.size();k++){
				Assignment a = new Assignment(ret); //or maybe include builder
				a.set(RV, RV.domain.get(k));
				double res = bn.getProb(RV, a);
				d.put(RV.domain.get(k), res);
			}
			d.normalize();
			double val = Math.random();
			//now test to see which value of the domain has been chosen
			double tot = 0.0;
			for(int i=0;i<RV.domain.size();i++){
				tot += d.get(RV.domain.get(i));
				if(val<= tot){
					ret.set(RV, RV.domain.get(i));
					break;
				}
			}
		}
		return ret;
	}
	
	//checks to ensure two assignments are consistent
	//that is that there contains no contradictions
	public boolean consistent(Assignment a,Assignment b){
		//loop through all random vars
		for(RandomVariable RV: a.keySet()){
			for(RandomVariable RV2 : b.keySet()){
				//if same name but different values contradiction found
				if((RV.getName()).equals(RV2.getName()) && !(a.get(RV)).equals(b.get(RV2))){
					return false;
				}
			}
		}
		//if no contradictions return true
		return true;
	}
}
