/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2015 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef SWC_SEGMENT_H
#define SWC_SEGMENT_H

/**
 *  This defines a single segment in an SWC file used in NeuroMorpho
 *  Note that this is not going to work well for dendritic spines which 
 *  come off in the middle of a parent compartment, and which have to be 
 *  started from the surface rather than the axis of the dendrite.
 */
class SwcSegment
{
	public:
		SwcSegment()
				: myIndex_( 0 ), type_( 0 ), radius_( 0.0 ),
				parent_( ~0U ), 
				geometricalDistanceFromSoma_( 0.0 ),
				electrotonicDistanceFromSoma_( 0.0 )
		{;}

		SwcSegment( const string& line );

		SwcSegment( int i,  unsigned short type, 
						double x, double y, double z, 
						double r, int parent );
		bool OK() const
		{
			return type_ < BadSegment;
		}

		void setBad() {
			type_ = BadSegment;
		}

		static unsigned short BadSegment;

		unsigned int parent() const {
			return parent_;
		}

		void setParent( unsigned int pa ) {
			parent_ = pa;
		}

		unsigned int myIndex() const {
			return myIndex_;
		}

		void addChild( unsigned int kid ) {
			kids_.push_back( kid );
		}

		void figureOutType();

		const vector< int >& kids() const
		{
			return kids_;
		}

		void replaceKids( const vector< int >& kids )
		{
			kids_ = kids;
		}

		double radius() const {
			return radius_;
		}

		unsigned short type() const {
			return type_;
		}

		double length( const SwcSegment& other ) const  {
			return v_.distance( other.v_ );
		}

		double L( const SwcSegment& other ) const;

		const Vec& vec() const {
			return v_;
		}

		void setCumulativeDistance( double rSoma, double eSoma )
		{
			geometricalDistanceFromSoma_ = rSoma;
			electrotonicDistanceFromSoma_ = eSoma;
		}
		
		double getGeomDistFromSoma() const  {
			return geometricalDistanceFromSoma_;
		}
		
		double getElecDistFromSoma() const  {
			return electrotonicDistanceFromSoma_;
		}

	protected:
		unsigned int myIndex_; /// Index of self
		/**
		 * The type of the segment is supposedly as below for SWC files.
		 * Seems to be honored in the breach in actual files.
		 * 0 = undefined
		 * 1 = soma
		 * 2 = axon
		 * 3 = dendrite
		 * 4 = apical dendrite
		 * 5 = fork point
		 * 6 = end point
		 * 7 = custom
		 */
		unsigned short type_; 
		Vec v_;	/// coordinates of end of segment
		double radius_; /// Radius of segment
		unsigned int parent_; /// Index of parent. Is ~0 for soma.

		/// dist from soma: not direct, but summed along dend
		double geometricalDistanceFromSoma_; 

		/// electrotonic dist from soma, summed along dend.
		double electrotonicDistanceFromSoma_; 

		vector< int > kids_; // Indices of all children of segment.
};

class SwcBranch: public SwcSegment
{
	public:
		SwcBranch( int i,  const SwcSegment& start, double len, double L,
				const vector< int >& cable );

		void printDiagnostics() const;

		double r0; /// Radius at beginning.
		double r1; /// Radius at end.

		/// Geometrical length of entire branch, summed along all segments.
		double geomLength;

		/**
		 * Electrotonic length summed along all branch segments. This does
		 * not include the assumed constant RA and RA terms, those are
		 * treated as 1.0. Suitable post-facto scaling needed.
		 */
		double electroLength;

		/**
		 * segs: ist of segments, in order away from soma. The starting
		 * entry is the one _after_ the fork point. The last entry is
		 * either a fork or an end point.
		 */
		vector< int > segs_; 
};

#endif // SWC_SEGMENT_H
