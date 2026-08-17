/* Advanced sorting code for TDR format data */
/* Produces ROOT histograms */

/* Tested up to ROOT 6.268/08 */

/* Changes to implement 4 detectors and generate event data */
/* based on time window */

/* Version 1 (base): (c) P.M. Jones (13/07/18)*/
/* Adapted:  (c) S. Hart (01/06/2023) */

/* This code reads in the list mode data from the buffer. It sorts the time and (rated) energy for each detector into a tree. */

/*   COMPILE & RUN: 
     clear && g++ -std=c++0x -O3 sort1labr.C -o exe `root-config --cflags --libs` -lSpectrum 
     ./exe ~/Documents/PhD/exp/2025/04032025/R17
*/
#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <inttypes.h> 
#include <time.h>
#include <cstdlib>
#include <pthread.h>

#include "TTree.h"
#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TAxis.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TF1.h"
#include "TSpectrum.h"
#include "TROOT.h"
#include "TSpectrum.h"
#include "TH3D.h"
#include "TH2F.h"
#include "TLatex.h"
#include "TRandom.h"
#include "TRandom3.h"

#define SIZE  16384
constexpr int kNumDetectors = 4;

static inline void compute_anger_position(
        const double energies[kNumDetectors],
        double &angerX,
        double &angerY,
        double &angerTotal)
{
    const double eA = energies[0];
    const double eB = energies[1];
    const double eC = energies[2];
    const double eD = energies[3];

    angerTotal = eA + eB + eC + eD;
    if (angerTotal > 0.0)
    {
        angerX = ((eB + eD) - (eA + eC)) / angerTotal;
        angerY = ((eC + eD) - (eA + eB)) / angerTotal;
    }
    else
    {
        angerX = NAN;
        angerY = NAN;
    }
}

int32_t buffer[SIZE];
uint64_t TStop, TSbot, data;
uint64_t TS, TSinit=0, TSfirst, counter,  SYNC, SYNClast=0;
uint64_t TimeStamp0[100]={0}, TimeStamp1[100]={0}, TimeStamp2[100]={0} ,TimeStamp3[100]={0}, TimeStamp4[100]={0};
uint16_t Energy[100]={0};
uint16_t QDC[100]={0}, qdc;
uint64_t TS2;
int64_t TSdiff, SYNCdiff;
int64_t count=0;
unsigned long int pos = 0;
int verbose = 255;

int main (int argc, char** argv)
{
  FILE *f;
  int i, j, k;
  int blocks_in=0;
  int twidset=0;
    std::string file_in;
    std::string file_out;
  int ident, card=0, adcdata, detectorID;
  int RunEnd=999999, Run=0;
  
  int m1,m2;
  int sum1, sum2;

  time_t start, end;
  time(&start);

  // Declare vectors to store the event data
    std::vector<double> energyS[kNumDetectors];
    std::vector<double> timeS[kNumDetectors];
  std::vector<int> pop(1, 0);
    double energySlow[kNumDetectors]={0,0,0,0}, timeSlow[kNumDetectors]={0,0,0,0};
    double angerX = NAN;
    double angerY = NAN;
    double angerTotal = NAN;
  double tdS;
  int Ed;

    // LaBr3 PSA pencil beam scan 
    std::vector<double> p1slow = {0.07273162941052977, 0.07274642204578287, 0.07262506063978541, 0.07108090891610476};
    std::vector<double> p0slow = {0, 0, 0, 0};

  double bw = 1, energy, rndchannel, rndUnif;
  TRandom3* randy = new TRandom3();
    
  int no_read;
  std::vector<int> time_stamp;
    
  if ((argc < 2) || (argc > 3))
  {
    fprintf(stderr, "Usage: %s data [last_subrun]\n", argv[0]);
    exit(1);
  }

  /* ROOT STUFF */
    file_out = std::string(argv[1]) + ".root";
  if (argc > 2) sscanf(argv[2], "%d", &RunEnd);
    TFile *g = TFile::Open(file_out.c_str(),"recreate");
    if (!g || g->IsZombie())
    {
        fprintf(stderr, "Failed to create ROOT file '%s'\n", file_out.c_str());
        return 1;
    }
    printf("ROOT file %s opened...\n", file_out.c_str());

  // ################ Create ROOT TTrees ################
  TTree *LaBrData = new TTree("LaBrData","LaBrData");
  LaBrData->Branch("slowEL0", &energySlow[0], "slowEL0/D");
  LaBrData->Branch("timeSL0", &timeSlow[0], "timeSL0/D");
  LaBrData->Branch("slowEL1", &energySlow[1], "slowEL1/D");
  LaBrData->Branch("timeSL1", &timeSlow[1], "timeSL1/D");
  LaBrData->Branch("slowEL2", &energySlow[2], "slowEL2/D");
  LaBrData->Branch("timeSL2", &timeSlow[2], "timeSL2/D");
  LaBrData->Branch("slowEL3", &energySlow[3], "slowEL3/D");
  LaBrData->Branch("timeSL3", &timeSlow[3], "timeSL3/D");
    LaBrData->Branch("angerX", &angerX, "angerX/D");
    LaBrData->Branch("angerY", &angerY, "angerY/D");
    LaBrData->Branch("angerTotal", &angerTotal, "angerTotal/D");


  // ################ Create histograms ################
    TH1D** slowE=new TH1D*[kNumDetectors];
    TH1D** slowTD0=new TH1D*[kNumDetectors];
    TH1D** slowTD1=new TH1D*[kNumDetectors];
        TH2D* angerXY = new TH2D("angerXY", "Anger centroid;X;Y", 200, -1.05, 1.05, 200, -1.05, 1.05);
        TH1D* angerTotalHist = new TH1D("angerTotalHist", "Anger total calibrated energy;Energy;Counts", 2000, 0, 2000);
    for (j = 0; j < kNumDetectors; j++)
  {
    slowE[j] = new TH1D(TString::Format("Slow_Energy_L%2d", j),"Spectrum",16380,0,16380);
    slowTD0[j] = new TH1D(TString::Format("Slow Time Difference_L0-L%2d", j),"Spectrum",10000,0,100); 
    slowTD1[j] = new TH1D(TString::Format("Slow Time Difference_L1-L%2d", j),"Spectrum",10000,0,100);
  }

  // ################ Start of data analysis ####################
  //File stuff
  while(Run < RunEnd+1)
   {
        file_in = std::string(argv[1]) + "_" + std::to_string(Run);
        
        f = fopen(file_in.c_str(), "rb");
        if (!f)
        {
        fprintf(stderr, "Can't open file '%s'\n", file_in.c_str());
        goto finish;
        }
        
        printf("File %s opened...\n", file_in.c_str());

        while (!feof(f))
        {
            //Read a whole block of data in (64kbytes)
        no_read = fread(buffer, sizeof(buffer[0]), SIZE, f);
        blocks_in++;
        
        if ( (feof(f)) && no_read <= 0 ) goto end;

        pos+=24;

            for (i=6; i< SIZE; i+=2)
            {
                //Start of new event data
                //Loop through each 64bit data work and decypher

                data = buffer[i+1];
                TSbot = buffer[i];
            
                if ((TSbot == 0) && (data == 0)) goto loop;
                if ((TSbot == 0x5e5e5e5e) && (data == 0x5e5e5e5e)) goto loop;
                if ((TSbot == 0x5e5e5e5e) && (data == 0xffffffff)) goto loop;
                
                /* DATA */
                if ((data & 0xc0000000) == 0xc0000000)
                {
                    ident = (data & 0x0fff0000) >> 16;
                    adcdata = (data & 0x0000ffff);
                    card = (ident / 32);
                    if (card == 99) goto skip;
                    TS = (TS & 0x0000fffff0000000ULL); // 100 MHz sampling speed
                    TS = ((TS | TSbot));
                    if (counter == 0)
                    {
                        TSfirst = TS;
                        counter = 1;
                    }
                    // TSdiff is the absolute value of TS-TSinit
                    TSdiff = TS-TSinit;

                    //std::cout << "TSinit " << TSinit << " TS " << TS << " TSdiff " << TSdiff << std::endl;
                    if ( (TSbot >= 0x0) && (TSbot <= 0x1a) )
                    {
                        if (twidset == 0) 
                        {
                        TS=TS+0x10000000;
                        twidset=1;
                        }          
                        TSdiff = TS-TSinit;  
                    }

                    /*
                    Channel Info:
                    ---------------------------------------------------------
                    Signal  |   slowT   |  slowE     |    
                    ---------------------------------------------------------
                    LaBr A  |    80     |    64      | 
                    ---------------------------------------------------------
                    LaBr B  |   81      |     65     |  
                    ---------------------------------------------------------
                    LaBr C  |   82      |     66     |
                    ---------------------------------------------------------
                    LaBr D  |   83      |     67     |
                    ---------------------------------------------------------
                    */

                    // Identify detector and store data accordingly
                    if ((ident > 63) && (ident < 68)) 
                    {
                        detectorID = ident-64;
                        if (energyS[detectorID].size() == 0)
                        {
                            double calibslowE = adcdata;
                            calibslowE = p1slow[detectorID]*pow(adcdata,1) + p0slow[detectorID];
                            rndUnif = randy->Uniform(-2,2);
                            energyS[detectorID].push_back(calibslowE+rndUnif);
                            //energyS[detectorID].push_back(adcdata);
                            //printf("energyS[%d] = %f\n", detectorID, energyS[detectorID][0]);
                            //printf("adcdata = %d\n\n", adcdata);
                        }

                    }
                    else if ((ident >79) && (ident < 84)) 
                    {
                        detectorID = ident-80;
                        int tick, tock;
                        double sick;
                        tock = ((adcdata & 0x0000e000) >> 13);
                        tick = (adcdata & 0x00001fff);
                        sick = (tick/8192.0) * 20.0; 
                        // if (tock != 7) // The time slow doesnt have the tock information -> there is no fractional range in the byte information between 0 and 7 for the 2ns sampling speed interval. Uncommenting this gives very low statistics.
                        {
                            if ((detectorID >= 0) && (detectorID < kNumDetectors) && timeS[detectorID].size() == 0) 
                            {
                                timeS[detectorID].push_back((TS * 100.0) + (tock * 20.0) + sick); // time in 1e12s which is 10ns
                                // printf("timeS[%d] = %f\n", detectorID, timeS[detectorID][0]);
                            }
                        }
                    }
                    

                    // printf("TSdiff %lld \n", TSdiff);
                    if (TSdiff >= 10 ) //90                     
                    {   
                        i=i-2; 
                        //std::cout <<"LOOP2 " << "i" << i << "| TSdiff " << TSdiff << "| TS " << TS << "| TSinit " << TSinit << std::endl;
                        for (j = 0; j < kNumDetectors; j++)
                        {
                            //printf("j %d | energyS[j].size() %lu | timeF[j].size() %lu | timeS[j].size() %lu \n", j, energyS[j].size(), timeF[j].size(), timeS[j].size());
                            if (energyS[j].size() == 0) energyS[j].push_back(0);
                            if (timeS[j].size() == 0)  timeS[j].push_back(0);

                            timeSlow[j] = timeS[j][0]/10;
                            energySlow[j] = energyS[j][0];

                            //printf("j %d | energySlow %f | timeSlow %f\n", j, energySlow[j],  timeSlow[j]);
                        }

                        compute_anger_position(energySlow, angerX, angerY, angerTotal);
                        if (std::isfinite(angerX) && std::isfinite(angerY))
                        {
                            angerXY->Fill(angerX, angerY);
                        }
                        if (std::isfinite(angerTotal))
                        {
                            angerTotalHist->Fill(angerTotal);
                        }

                       // printf("timeF %f | timeS %f | energyS %f\n", timeFast[0]  , timeSlow[0]  , energySlow[0]);
                        LaBrData->Fill();

                        for (j = 0; j < kNumDetectors; j++) 
                        { 
                            //Singles spectra 
                            //std::cout << "L" << j << "| energySlow[j] " << energySlow[j] << "| timeFast[j] " << timeFast[j] << "| timeSlow[j] " << timeSlow[j] << "| timeRF" << timeFast[4]<< std::endl;
                            slowE[j]->Fill(energySlow[j]);

                            for (k = 0; k < kNumDetectors; k++) 
                            {
                                if (j != k ) 
                                {
                                    tdS = (timeSlow[j]-timeSlow[k]);
                                    if (j == 1) 
                                    {
                                        // print the slowEnergy[0] and slowEnergy[1] values
                                        //printf("slowE[0] %f | slowE[1] %f\n", energySlow[0], energySlow[1]);
                                        if (energySlow[0] > 662-10.35 && energySlow[0] < 662+10.35) // 1173 keV peak
                                        {
                                            if (energySlow[1] > 662-10.35 && energySlow[1] < 662+10.35) // 1332 keV peak
                                            {
                                                /* slowTD1[k]->Fill(tdS);
                                                slowTD0[j]->Fill(tdS); */
                                            }
                                        }
                                    }
                                }
                            }                        
                        }

                        // Reset vectors
                        for (j = 0; j < kNumDetectors; j++) 
                        {
                            energyS[j].clear();
                            timeS[j].clear();
                            energySlow[j] = 0;
                            timeSlow[j] = 0;
                        }
                        
                    }
                    TSinit=TS;
                }

                /* SYNC */
                if ((data & 0xc0f00000) == 0x80400000) 
                {
                    card = (data & 0x3f000000) >> 24;
                    TStop = (data & 0x000fffff);
                    TS = (TStop);
                    TS = ((TS << 28));
                    TS = ((TS | TSbot));

                    TSdiff = TS-TSinit;

                    SYNC = TS;
                    SYNCdiff = SYNC-SYNClast;

                    twidset=0;

                    SYNClast = SYNC;

                }

                
                skip:
                count++;
                loop:
                pos+=8; // 8 bytes per event
            }
        }
        end:  
        //std::cout << "End of file reached" << std::endl;
        fclose(f);
        
        Run++;

    }
    finish:
    

    LaBrData->Write();

    for (j = 0; j < 4; j++)
    {
        slowE[j]->Write();
        slowTD0[j]->Write();
        slowTD1[j]->Write();
    }
    angerXY->Write();
    angerTotalHist->Write();


    // Calculating total time taken by the program.
    double TimeDiff = (double) (TS - TSfirst)*(1e-8)/60.0;
    printf("\nThe first time stamp is (e-8 s): %" PRIu64 "\n", TSfirst);
    printf("The last time stamp is (e-8 s): %" PRIu64 "\n", TS);
    printf("The time difference of the RXX_ file is (min): %f\n", TimeDiff);

    time(&end);

    double time_taken = double((end - start));
    std::cout << "\nTime taken by program is : " << time_taken << std::setprecision(5);
    std::cout << " seconds " << std::endl;

    
    delete g;
}