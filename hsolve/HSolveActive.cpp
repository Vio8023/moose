/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include <queue>
#include "HSolveStruct.h"
#include "HinesMatrix.h"
#include "HSolvePassive.h"
#include "HSolveActive.h"
#include "HSolve.h"
#include "../biophysics/CompartmentBase.h"
#include "../biophysics/Compartment.h"
#include "../biophysics/CaConcBase.h"
#include "ZombieCaConc.h"

#include "CudaGlobal.h"
#include "RateLookup.h"

#ifdef USE_CUDA
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

#include <sys/time.h>

typedef unsigned long long u64;

/* get microseconds (us) */
u64 getTime()
{
 struct timeval tv;

 gettimeofday(&tv, NULL);

 u64 ret = tv.tv_usec;

 ret += (tv.tv_sec * 1000 * 1000);

 return ret;
}

using namespace moose;
using namespace thrust;

//~ #include "ZombieCompartment.h"
//~ #include "ZombieCaConc.h"

extern ostream& operator <<( ostream& s, const HinesMatrix& m );

const int HSolveActive::INSTANT_X = 1;
const int HSolveActive::INSTANT_Y = 2;
const int HSolveActive::INSTANT_Z = 4;

HSolveActive::HSolveActive()
{
    caAdvance_ = 1;
    
#ifdef USE_CUDA    
	current_ca_position = 0;
	is_inited_ = 0;
	for(int i = 0; i < 10; ++i)
	{
		total_time[i] = 0;
	}
	total_count = 0;
#endif

    // Default lookup table size
    //~ vDiv_ = 3000;    // for voltage
    //~ caDiv_ = 3000;   // for calcium
}

//////////////////////////////////////////////////////////////////////
// Solving differential equations
//////////////////////////////////////////////////////////////////////
void update_info(double* time, int func, int elapsed, int count, double dt)
{
	time[func] += elapsed / 1000.0f;
	char * str;
	if(count >= (0.2/dt) - 1)
	{
		switch(func)
		{
			case 0: str = "advanceChannels";break;
			case 1: str = "calculateChannelCurrents";break;
			case 2: str = "updateMatrix";break;
			case 3: str = "forwardEliminate";break;
			case 4: str = "backwardSubstitute";break;
			case 5: str = "advanceCalcium";break;
			case 6: str = "advanceSynChans";break;
			case 7: str = "sendValues";break;
			case 8: str = "sendSpikes";break;
			case 9: str = "externalCurrent_.assign";break;
			default: str = "Unkown";break;
		}
		printf("Function %s takes %fs.\n", str, time[func]/ 1000.0f);
	}
}
void HSolveActive::step( ProcPtr info )
{	
    if ( nCompt_ <= 0 )
        return;

    if ( !current_.size() )
    {
        current_.resize( channel_.size() );
    }
    
    total_count ++;
    
	  u64 start_time, end_time;
    start_time = getTime();
    
    advanceChannels( info->dt );
    
    end_time = getTime();
    update_info(total_time, 0, end_time - start_time, total_count ,info->dt);
    start_time = end_time;
    
    calculateChannelCurrents();
    
    end_time = getTime();
    update_info(total_time, 1, end_time - start_time, total_count ,info->dt);
    start_time = end_time;   
    
    updateMatrix();
    end_time = getTime();
    update_info(total_time, 2, end_time - start_time, total_count ,info->dt);
    start_time = end_time;   
        
    HSolvePassive::forwardEliminate();
    
    end_time = getTime();
    update_info(total_time, 3, end_time - start_time, total_count ,info->dt);
    start_time = end_time;   
        
    HSolvePassive::backwardSubstitute();
    
    end_time = getTime();
    update_info(total_time, 4, end_time - start_time, total_count ,info->dt);
    start_time = end_time;   
        
    advanceCalcium();
    
    end_time = getTime();
    update_info(total_time, 5, end_time - start_time, total_count ,info->dt);
    start_time = end_time;   
        
    advanceSynChans( info );
    
    end_time = getTime();
    update_info(total_time, 6, end_time - start_time, total_count ,info->dt);
    start_time = end_time;   
    
    sendValues( info );
    
    end_time = getTime();
    update_info(total_time, 7, end_time - start_time, total_count ,info->dt);
    start_time = end_time;   
        
    sendSpikes( info );
     
    end_time = getTime();
    update_info(total_time, 8, end_time - start_time, total_count ,info->dt);
    start_time = end_time;   
       

    externalCurrent_.assign( externalCurrent_.size(), 0.0 );
    
    end_time = getTime();
    update_info(total_time, 9, end_time - start_time, total_count ,info->dt);
}

void HSolveActive::calculateChannelCurrents()
{
    vector< ChannelStruct >::iterator ichan;
    vector< CurrentStruct >::iterator icurrent = current_.begin();

    if ( state_.size() != 0 )
    {
        double* istate = &state_[ 0 ];

        for ( ichan = channel_.begin(); ichan != channel_.end(); ++ichan )
        {
            ichan->process( istate, *icurrent );
            ++icurrent;
        }
    }
}

void HSolveActive::updateMatrix()
{
    /*
     * Copy contents of HJCopy_ into HJ_. Cannot do a vector assign() because
     * iterators to HJ_ get invalidated in MS VC++
     */
    if ( HJ_.size() != 0 )
        memcpy( &HJ_[ 0 ], &HJCopy_[ 0 ], sizeof( double ) * HJ_.size() );

    double GkSum, GkEkSum; vector< CurrentStruct >::iterator icurrent = current_.begin();
    vector< currentVecIter >::iterator iboundary = currentBoundary_.begin();
    vector< double >::iterator ihs = HS_.begin();
    vector< double >::iterator iv = V_.begin();

    vector< CompartmentStruct >::iterator ic;
    for ( ic = compartment_.begin(); ic != compartment_.end(); ++ic )
    {
        GkSum   = 0.0;
        GkEkSum = 0.0;
        for ( ; icurrent < *iboundary; ++icurrent )
        {
            GkSum   += icurrent->Gk;
            GkEkSum += icurrent->Gk * icurrent->Ek;
        }

        *ihs = *( 2 + ihs ) + GkSum;
        *( 3 + ihs ) = *iv * ic->CmByDt + ic->EmByRm + GkEkSum;

        ++iboundary, ihs += 4, ++iv;
    }

    map< unsigned int, InjectStruct >::iterator inject;
    for ( inject = inject_.begin(); inject != inject_.end(); ++inject )
    {
        unsigned int ic = inject->first;
        InjectStruct& value = inject->second;

        HS_[ 4 * ic + 3 ] += value.injectVarying + value.injectBasal;

        value.injectVarying = 0.0;
    }

    // Synapses are being handled as external channels.
    //~ double Gk, Ek;
    //~ vector< SynChanStruct >::iterator isyn;
    //~ for ( isyn = synchan_.begin(); isyn != synchan_.end(); ++isyn ) {
    //~ get< double >( isyn->elm_, synGkFinfo, Gk );
    //~ get< double >( isyn->elm_, synEkFinfo, Ek );
    //~
    //~ unsigned int ic = isyn->compt_;
    //~ HS_[ 4 * ic ] += Gk;
    //~ HS_[ 4 * ic + 3 ] += Gk * Ek;
    //~ }

    ihs = HS_.begin();
    vector< double >::iterator iec;
    for ( iec = externalCurrent_.begin(); iec != externalCurrent_.end(); iec += 2 )
    {
        *ihs += *iec;
        *( 3 + ihs ) += *( iec + 1 );

        ihs += 4;
    }

    stage_ = 0;    // Update done.
}

void HSolveActive::advanceCalcium()
{
    vector< double* >::iterator icatarget = caTarget_.begin();
    vector< double >::iterator ivmid = VMid_.begin();
    vector< CurrentStruct >::iterator icurrent = current_.begin();
    vector< currentVecIter >::iterator iboundary = currentBoundary_.begin();

    /*
     * caAdvance_: This flag determines how current flowing into a calcium pool
     * is computed. A value of 0 means that the membrane potential at the
     * beginning of the time-step is used for the calculation. This is how
     * GENESIS does its computations. A value of 1 means the membrane potential
     * at the middle of the time-step is used. This is the correct way of
     * integration, and is the default way.
     */
    if ( caAdvance_ == 1 )
    {
        for ( ; iboundary != currentBoundary_.end(); ++iboundary )
        {
            for ( ; icurrent < *iboundary; ++icurrent )
            {
                if ( *icatarget )
                    **icatarget += icurrent->Gk * ( icurrent->Ek - *ivmid );

                ++icatarget;
            }

            ++ivmid;
        }
    }
    else if ( caAdvance_ == 0 )
    {
        vector< double >::iterator iv = V_.begin();
        double v0;

        for ( ; iboundary != currentBoundary_.end(); ++iboundary )
        {
            for ( ; icurrent < *iboundary; ++icurrent )
            {
                if ( *icatarget )
                {
                    v0 = ( 2 * *ivmid - *iv );

                    **icatarget += icurrent->Gk * ( icurrent->Ek - v0 );
                }

                ++icatarget;
            }

            ++ivmid, ++iv;
        }
    }

    vector< CaConcStruct >::iterator icaconc;
    vector< double >::iterator icaactivation = caActivation_.begin();
    vector< double >::iterator ica = ca_.begin();
    for ( icaconc = caConc_.begin(); icaconc != caConc_.end(); ++icaconc )
    {
        *ica = icaconc->process( *icaactivation );
        ++ica, ++icaactivation;
    }

    caActivation_.assign( caActivation_.size(), 0.0 );
}

void HSolveActive::advanceChannels( double dt )
{

    vector< double >::iterator iv;
    vector< double >::iterator istate = state_.begin();
    vector< int >::iterator ichannelcount = channelCount_.begin();
    vector< ChannelStruct >::iterator ichan = channel_.begin();
    vector< ChannelStruct >::iterator chanBoundary;
    vector< unsigned int >::iterator icacount = caCount_.begin();
    vector< double >::iterator ica = ca_.begin();
    vector< double >::iterator caBoundary;
    vector< LookupColumn >::iterator icolumn = column_.begin();
    vector< LookupRow >::iterator icarowcompt;
    vector< LookupRow* >::iterator icarow = caRow_.begin();

    LookupRow vRow;
#ifdef USE_CUDA
#ifdef DEBUG_STEP
    printf("Press [ENTER] to start advanceChannels...\n");
    getchar();
#endif    

    vector<double> caRow_ac;
    vector<LookupColumn> column_ac;
    
    iv = V_.begin();

    double * v_row_array_d;

    if(V_.size() < 1024 && 0)
    {
        vector<double> v_row_temp(V_.size());
        vector<double>::iterator v_row_iter = v_row_temp.begin();
        for(u32 i = 0 ; i < V_.size(); ++i)
        {
            vTable_.row(*iv, *v_row_iter);
            iv++;
            v_row_iter++;
        }       

        copy_to_device(&v_row_array_d, &v_row_temp.front(), V_.size());

    } else {
        vTable_.row_gpu(iv, &v_row_array_d, V_.size());
    }
#if defined(DEBUG_) && defined(DEBUG_VERBOSE) 
    printf("Trying to access v_row_array_d...\n");
    
    std::vector<double> h_row(V_.size());
    cudaSafeCall(cudaMemcpy(&h_row.front(), v_row_array_d, sizeof(double) * V_.size(), cudaMemcpyDeviceToHost));
    printf("row 0 is %f.\n", h_row[0]);
    printf("last row is %f.\n", h_row[V_.size() - 1]);
#ifdef DEBUG_STEP
    getchar();
#endif    
#endif  


#if defined(DEBUG_) && defined(DEBUG_VERBOSE) 
    printf("Starting converting caRow_ to caRow_ac...\n");
#ifdef DEBUG_STEP
    getchar();
#endif    
#endif 
    
    caRow_ac.resize(caRow_.size());
    for(u32 i = 0; i < caRow_.size(); ++i)
    {
        if(caRow_[i])
        {
            caRow_ac[i] = caRow_[i]->rowIndex + caRow_[i]->fraction;
        } 
        else
        {
            caRow_ac[i] = -1.0f;
        } 
       
    }

#if defined(DEBUG_) && defined(DEBUG_VERBOSE)   
    printf("Starting find-row for caRowCompt_ and vRow_ac construction...\n");
#ifdef DEBUG_STEP
    getchar();
#endif    
#endif

    for (int i = 0; i < V_.size(); ++i) {
        icarowcompt = caRowCompt_.begin();
        caBoundary = ica + *icacount;
        
        for ( ; ica < caBoundary; ++ica )
        {
            caTable_.row( *ica, * icarowcompt );
            ++icarowcompt;
        }
        
        ++icacount;
    }

#if defined(DEBUG_) && defined(DEBUG_VERBOSE)  
    printf("Finish preparing CUDA advanceChannel! \n");
    printf("Starting kernel...\n");
#ifdef DEBUG_STEP
    getchar();
#endif    
#endif    
    copy_data(column_,
    		  &column_d,
    		  &is_inited_,
    		  channel_data_,
    		  &channel_data_d,
    		  HSolveActive::INSTANT_X,
              HSolveActive::INSTANT_Y,
              HSolveActive::INSTANT_Z);

    advanceChannel_gpu(v_row_array_d, 
                       caRow_ac, 
                       column_d, 
                       vTable_, 
                       caTable_, 
                       &state_.front(), 
                       channel_data_d,
                       dt,
                       (int)(column_.size()),
                       (int)(channel_data_.size()),
                       V_.size());

    caRow_ac.clear();

#if defined(DEBUG_) && defined(DEBUG_VERBOSE)  
    printf("Finish launching CUDA advanceChannel! \n");
#ifdef DEBUG_STEP
    getchar();
#endif    
#endif 
#else    
    double C1, C2;

    for ( iv = V_.begin(); iv != V_.end(); ++iv )
    {
        vTable_.row( *iv, vRow );
        icarowcompt = caRowCompt_.begin();
        caBoundary = ica + *icacount;
        
        for ( ; ica < caBoundary; ++ica )
        {
            caTable_.row( *ica, * icarowcompt );
            ++icarowcompt;
        }   
        /*
         * Optimize by moving "if ( instant )" outside the loop, because it is
         * rarely used. May also be able to avoid "if ( power )".
         *
         * Or not: excellent branch predictors these days.
         *
         * Will be nice to test these optimizations.
         */
        chanBoundary = ichan + *ichannelcount;
        for ( ; ichan < chanBoundary; ++ichan )
        {
            if ( ichan->Xpower_ > 0.0 )
            {
                vTable_.lookup( *icolumn, vRow, C1, C2 );
                //~ *istate = *istate * C1 + C2;
                //~ *istate = ( C1 + ( 2 - C2 ) * *istate ) / C2;
                if ( ichan->instant_ & INSTANT_X )
                    *istate = C1 / C2;
                else
                {
                    double temp = 1.0 + dt / 2.0 * C2;
                    *istate = ( *istate * ( 2.0 - temp ) + dt * C1 ) / temp;
                }

                ++icolumn, ++istate;
            }

            if ( ichan->Ypower_ > 0.0 )
            {
                vTable_.lookup( *icolumn, vRow, C1, C2 );
                //~ *istate = *istate * C1 + C2;
                //~ *istate = ( C1 + ( 2 - C2 ) * *istate ) / C2;
                if ( ichan->instant_ & INSTANT_Y )
                    *istate = C1 / C2;
                else
                {
                    double temp = 1.0 + dt / 2.0 * C2;
                    *istate = ( *istate * ( 2.0 - temp ) + dt * C1 ) / temp;
                }

                ++icolumn, ++istate;
            }

            if ( ichan->Zpower_ > 0.0 )
            {
                LookupRow* caRow = *icarow;
                if ( caRow )
                {
                    caTable_.lookup( *icolumn, *caRow, C1, C2 );
                }
                else
                {
                    vTable_.lookup( *icolumn, vRow, C1, C2 );
                }

                //~ *istate = *istate * C1 + C2;
                //~ *istate = ( C1 + ( 2 - C2 ) * *istate ) / C2;
                if ( ichan->instant_ & INSTANT_Z )
                    *istate = C1 / C2;
                else
                {
                    double temp = 1.0 + dt / 2.0 * C2;
                    *istate = ( *istate * ( 2.0 - temp ) + dt * C1 ) / temp;
                }

                ++icolumn, ++istate, ++icarow;
            }
        }

        ++ichannelcount, ++icacount;
    }
#endif
      
}
LookupColumn * HSolveActive::get_column_d()
{
	return column_d;
}

/**
 * SynChans are currently not under solver's control
 */
void HSolveActive::advanceSynChans( ProcPtr info )
{
    return;
}

void HSolveActive::sendSpikes( ProcPtr info )
{
    vector< SpikeGenStruct >::iterator ispike;
    for ( ispike = spikegen_.begin(); ispike != spikegen_.end(); ++ispike )
        ispike->send( info );
}

/**
 * This function dispatches state values via any source messages on biophysical
 * objects which have been taken over.
 *
 */
void HSolveActive::sendValues( ProcPtr info )
{
    vector< unsigned int >::iterator i;

    for ( i = outVm_.begin(); i != outVm_.end(); ++i )
        moose::Compartment::VmOut()->send(
            //~ ZombieCompartment::VmOut()->send(
            compartmentId_[ *i ].eref(),
            V_[ *i ]
        );

    for ( i = outCa_.begin(); i != outCa_.end(); ++i )
        //~ CaConc::concOut()->send(
        CaConcBase::concOut()->send(
            caConcId_[ *i ].eref(),
            ca_[ *i ]
        );
}
