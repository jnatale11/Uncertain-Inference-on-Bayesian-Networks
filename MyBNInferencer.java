import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import javax.xml.parsers.ParserConfigurationException;

import org.xml.sax.SAXException;

//Jason Natale
//CSC242 Project 3
//Inferencer makes calculations on posterior distributions
public class MyBNInferencer{
	//called with filename, query variable, and evidence variables with assignment
	public static void main(String[] args){
		//get info from commands
		String filename = args[0];
		String queryvarname = "";
		if(args.length>=2)
			queryvarname = args[1];
		int len = args.length;
		int numEvid = (len -2)/2;
		String[] enames = new String[numEvid];
		String[] evals = new String[numEvid];
		//at each loop take a new evidence var
		for(int i=0;i<numEvid;i++){
			enames[i] = args[2+i*2];
			evals[i] = args[3+i*2];
		}
		//either xml or bif
		String end = filename.substring(filename.length()-3);
		boolean xml = false;
		if(end.equals("xml"))
			xml = true;
		//convert file information into bayesian network
		BayesianNetwork bae = new BayesianNetwork();
		if(xml){
			XMLBIFParser parser = new XMLBIFParser();
			try{
				bae = parser.readNetworkFromFile(filename);
			}
			//in case file cannot be read
			catch(IOException e){error(1,e.getMessage());}
			catch(SAXException e){error(2,e.getMessage());}
			catch(ParserConfigurationException e){error(3,e.getMessage());}
		}
		else{
			try{
				//working with .bif
			FileInputStream fis = new FileInputStream(filename);
			BIFLexer lex = new BIFLexer(fis);
			BIFParser parser = new BIFParser(lex);
			bae = parser.parseNetwork();
			}
			catch(IOException e){
				System.out.println("Error reading .bif");
			}
		}
		//get possible assignments of query var
		RandomVariable qv = bae.getVariableByName(queryvarname);
		//make assignments
		Assignment assign = new Assignment();
		for(int a=0;a<numEvid;a++){
			assign.set(bae.getVariableByName(enames[a]), evals[a]);
		}
		//finally get distribution
		MyBNInferencer tester = new MyBNInferencer();
		double t = System.currentTimeMillis();
		Distribution distrib = tester.ask(bae,qv,assign);
		System.out.println("Time taken: "+(System.currentTimeMillis()-t)/1000);
		//output distribution
		System.out.println(distrib.toString());
	}
	
	//creates class object 
	public MyBNInferencer(){}
	
	//delivers error message
	public static void error(int a, String msg){
		System.out.println("error making BN "+msg);
	}
	
	//based off of pseudocode from Figure 14.9 of AIMA
	//returns distribution of a probabilities for a given query variable 
	//in a bayesian network - Enumeration Algorithm
	public Distribution ask(BayesianNetwork bn, RandomVariable X, Assignment e){
		Distribution d = new Distribution(X);
		//start putting values into d after calculating each value's probability
		//looping by number of potential values
		for(int i=0;i<X.domain.size();i++){
			//make new assignment with X set to one if its domain vals
			Assignment newa = new Assignment(e);
			newa.set(X, X.domain.get(i));
			//get list of variables in bn
			d.put(X.domain.get(i),Enumerate_All(bn,bn.getVariableListTopologicallySorted(),newa));
		}
		//normalize d before returning
		d.normalize();
		return d;
	}
	
	//ENUMERATE-ALL Method as described in Figure 14.9 of AIMA
	//works recursively to backtrack a network and solve probabilities
	public double Enumerate_All(BayesianNetwork bn, List<RandomVariable> vars, Assignment e){
		//make new temporary list variable for RVs
		List<RandomVariable> Vars = new ArrayList<RandomVariable>();
		for(int i=0;i<vars.size();i++)
			Vars.add(vars.get(i));
		//when reaching end of vars, return 1 to be multiplied by
		if(Vars.size() == 0)
			return 1.0;
		//otherwise take first variable off of list
		RandomVariable Y = Vars.get(0);
		//take var out
		Vars.remove(Y);
		//if Y is mentioned in the assignment
		if(e.containsKey(Y)){
			//calculate conditional probability of that value occuring, times recursive call for more vars
			double furtherProb = Enumerate_All(bn,Vars,e);
			double getProb = bn.getProb(Y, e);
			return getProb * furtherProb;
		}
		else{
			double ret = 0.0;
			//loop through each possible assignment of Y (domain of Y)
			for(int k=0;k<Y.domain.size();k++){
				//make new assignment for each value of Y
				Assignment newone = new Assignment(e);
				newone.set(Y, Y.domain.get(k));
				ret += (bn.getProb(Y, newone) * Enumerate_All(bn,Vars,newone));
			}
			return ret;
		}
	}
}
