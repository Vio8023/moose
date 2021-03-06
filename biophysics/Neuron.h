/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _NEURON_H
#define _NEURON_H

/**
 * The Neuron class to hold the Compartment class elements.
 */

class Neuron
{
	public:
		Neuron();
		void setRM( double v );
		double getRM() const;
		void setRA( double v );
		double getRA() const;
		void setCM( double v );
		double getCM() const;
		void setEm( double v );
		double getEm() const;
		void setTheta( double v );
		double getTheta() const;
		void setPhi( double v );
		double getPhi() const;
		void setSourceFile( string v );
		string getSourceFile() const;
		void setCompartmentLengthInLambdas( double v );
		double getCompartmentLengthInLambdas() const;
		unsigned int getNumCompartments() const;
		unsigned int getNumBranches() const;
		vector< double> getGeomDistFromSoma() const;
		vector< double> getElecDistFromSoma() const;
		void setChannelDistribution( vector< string > v );
		vector< string > getChannelDistribution() const;

		void buildSegmentTree( const Eref& e );
		void assignChanDistrib( const Eref& e,
			string name, string path, string func );
		void clearChanDistrib( const Eref& e,
			string name, string path );
		void parseChanDistrib( const Eref& e );
		/**
		 * Initializes the class info.
		 */
		static const Cinfo* initCinfo();
	private:
		double RM_;
		double RA_;
		double CM_;
		double Em_;
		double theta_;
		double phi_;
		string sourceFile_;
		double compartmentLengthInLambdas_;
		vector< string > channelDistribution_;
		vector< Id > segId_; /// Id of each Seg entry, below.
		vector< SwcSegment > segs_;
		vector< SwcBranch > branches_;

};

// 

#endif // 
