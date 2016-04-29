/***
 *       Filename:  Streamer.cpp
 *
 *    Description:  Stream table data.
 *
 *        Version:  0.0.1
 *        Created:  2016-04-26

 *       Revision:  none
 *
 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *
 *        License:  GNU GPL2
 */

#include <algorithm>
#include <sstream>

#include "global.h"
#include "header.h"
#include "Streamer.h"
#include "Clock.h"

const Cinfo* Streamer::initCinfo()
{
    /*-----------------------------------------------------------------------------
     * Finfos
     *-----------------------------------------------------------------------------*/
    static ValueFinfo< Streamer, string > outfile(
        "outfile"
        , "File/stream to write table data to. Default is is tables.dat."
        , &Streamer::setOutFilepath
        , &Streamer::getOutFilepath
    );

    static ValueFinfo< Streamer, string > format(
        "format"
        , "Format of output file, default is csv"
        , &Streamer::setFormat
        , &Streamer::getFormat
    );

    static ReadOnlyValueFinfo< Streamer, size_t > numTables (
        "numTables"
        , "Number of Tables handled by Streamer "
        , &Streamer::getNumTables
    );

    /*-----------------------------------------------------------------------------
     *
     *-----------------------------------------------------------------------------*/
    static DestFinfo process(
        "process"
        , "Handle process call"
        , new ProcOpFunc< Streamer >( &Streamer::process )
    );

    static DestFinfo reinit(
        "reinit"
        , "Handles reinit call"
        , new ProcOpFunc< Streamer > ( &Streamer::reinit )
    );


    static DestFinfo addTable(
        "addTable"
        , "Add a table to Streamer"
        , new OpFunc1<Streamer, Id>( &Streamer::addTable )
    );

    static DestFinfo addTables(
        "addTables"
        , "Add many tables to Streamer"
        , new OpFunc1<Streamer, vector<Id> >( &Streamer::addTables )
    );

    static DestFinfo removeTable(
        "removeTable"
        , "Remove a table from Streamer"
        , new OpFunc1<Streamer, Id>( &Streamer::removeTable )
    );

    static DestFinfo removeTables(
        "removeTables"
        , "Remove tables -- if found -- from Streamer"
        , new OpFunc1<Streamer, vector<Id> >( &Streamer::removeTables )
    );

    /*-----------------------------------------------------------------------------
     *  ShareMsg definitions.
     *-----------------------------------------------------------------------------*/
    static Finfo* procShared[] =
    {
        &process , &reinit , &addTable, &addTables, &removeTable, &removeTables
    };

    static SharedFinfo proc(
        "proc",
        "Shared message for process and reinit",
        procShared, sizeof( procShared ) / sizeof( const Finfo* )
    );

    static Finfo * tableStreamFinfos[] =
    {
        &outfile, &format, &proc, &numTables
    };

    static string doc[] =
    {
        "Name", "Streamer",
        "Author", "Dilawar Singh, 2016, NCBS, Bangalore.",
        "Description", "Streamer: Stream moose.Table data to out-streams\n"
    };

    static Dinfo< Streamer > dinfo;

    static Cinfo tableStreamCinfo(
        "Streamer",
        TableBase::initCinfo(),
        tableStreamFinfos,
        sizeof( tableStreamFinfos )/sizeof(Finfo *),
        &dinfo,
        doc,
        sizeof(doc) / sizeof(string)
    );

    return &tableStreamCinfo;
}

static const Cinfo* tableStreamCinfo = Streamer::initCinfo();

// Class function definitions

Streamer::Streamer() : delimiter_( ","), format_( "csv" )
{
}

Streamer& Streamer::operator=( const Streamer& st )
{
    return *this;
}


Streamer::~Streamer()
{
}

/**
 * @brief Reinit.
 *
 * @param e
 * @param p
 */
void Streamer::reinit(const Eref& e, ProcPtr p)
{
    // If it is not stdout, then open a file and write standard header to it.
    initOutfile( e );
}

/**
 * @brief This function is called at its clock tick.
 *
 * @param e
 * @param p
 */
void Streamer::process(const Eref& e, ProcPtr p)
{


    writeTablesToOutfile( );
}


/**
 * @brief Add a table to streamer.
 *
 * @param table Id of table.
 */
void Streamer::addTable( Id table )
{
    // If this table is not already in the vector, add it.
    for( auto t : tables_ )
        if( table.path() == t.first.path() )
            return;                             /* Already added. */

    TableBase* t = reinterpret_cast<TableBase*>(table.eref().data());
    tables_[ table ] = t;
}

/**
 * @brief Add multiple tables to Streamer.
 *
 * @param tables
 */
void Streamer::addTables( vector<Id> tables )
{
    for( auto t : tables ) addTable( t );
}


/**
 * @brief Remove a table from Streamer.
 *
 * @param table. Id of table.
 */
void Streamer::removeTable( Id table )
{
    auto it = tables_.find( table );
    if( it != tables_.end() )
        tables_.erase( it );
}

/**
 * @brief Remove multiple tables -- if found -- from Streamer.
 *
 * @param tables
 */
void Streamer::removeTables( vector<Id> tables )
{
    for( auto t : tables ) removeTable( t );
}

/**
 * @brief Get the number of tables handled by Streamer.
 *
 * @return  Number of tables.
 */
size_t Streamer::getNumTables( void ) const
{
    return tables_.size();
}

/**
 * @brief Write given string to text file and clear it.
 *
 * @param text
 */
void Streamer::write( string& text )
{
    of_ << text;
    text = "";
}


void Streamer::initOutfile( const Eref& e )
{
    if( ! of_.is_open() )
        std::cerr << "Warn: Could not open file " << outfilePath_
                  << ". I am going to write to 'tables.dat'. "
                  << endl;

    // Now write header to this file. First column is always time
    text_ = "time" + delimiter_;
    for( auto t : tables_ )
        text_ += t.first.path() + delimiter_;
    // Remove the last command and add newline.
    text_.pop_back(); text_ += '\n';

    // Write to stream.
    write( text_ );

    // Initialize the clock and it dt.
    int numTick = e.element()->getTick();
    Clock* clk = reinterpret_cast<Clock*>(Id(1).eref().data());
    dt_ = clk->getTickDt( numTick );
}


string Streamer::getOutFilepath( void ) const
{
    return outfilePath_;
}

void Streamer::setOutFilepath( string filepath )
{
    outfilePath_ = filepath;
}

/* Set the format of all Tables */
void Streamer::setFormat( string format )
{
    format_ = format;
}

/*  Get the format of all tables. */
string Streamer::getFormat( void ) const 
{
    return format_;
}

/**
 * @brief Write data of its table to output file.
 */
void Streamer::writeTablesToOutfile( void )
{
    if( tables_.size() <= 0 )
        return;

    vector<vector<double> > data( tables_.size() );
    vector<size_t> dataSize( tables_.size() );

    size_t i = 0;
    for( auto tab : tables_ )
    {
        dataSize[i] = tab.second->getVecSize();

        // If any table has fewer data points then the threshold for writing to
        // file then return without doing anything.
        data[i] = tab.second->getVec();

        // Clear the data from vector
        tab.second->clearVec();

        i++;
    }

    if( std::min_element( dataSize.begin(), dataSize.end() ) !=
            std::max_element( dataSize.begin(), dataSize.end() )
      )
    {
        cout << "WARNING: One or more tables handled by this Streamer are collecting "
             << "data at different rate than others. I'll continue dumping data to "
             << "stream/file but it will get corrupted. I'll advise you to delete  "
             << "such tables."
             << endl;
    }

    // All vectors must be of same size otherwise we are in trouble.
    for (size_t i = 0; i < dataSize[0]; i++)
    {
        text_ += moose::global::toString<double>(dt_ * numLinesWritten_) + delimiter_;
        for (size_t ii = 0; ii < data.size(); ii++)
            text_ += moose::global::toString<double>(data[ii][i]) + delimiter_;
        // Remove last "," and append a new line.
        text_.pop_back(); text_ += '\n';
        numLinesWritten_ += 1;
    }

    write( text_ );
}


