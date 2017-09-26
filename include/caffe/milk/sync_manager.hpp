#ifndef __SYNC_MANAGER_H__
#define __SYNC_MANAGER_H__

#include <iostream>
#include <vector>
#include <map>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

using caffe::Blob;
using caffe::Caffe;

#define CLOSE_CONN -1
#define PUSH_FULL_MODEL -2
#define PUSH_A_DATA -3
#define PULL_FULL_MODEL -4
#define PULL_A_DATA -5
#define PUSH_FULL_DIFF -6
#define PUSH_A_DIFF -7
#define PULL_FULL_DIFF -8
#define PULL_A_DIFF -9
#define CLOSE_SIG -10

#define SERVER_MODE 0
#define CLIENT_MODE 1

template<typename Dtype>
class SyncManager {
    public:
        SyncManager(int mode_, int num, const std::vector<Blob<Dtype>*>& learnable_params_);
        ~SyncManager();

        // for client
        void pushDiff();
        void pullModel();
        // for server
        void getDiff(int sock);
        void sendModel(int sock);
        void waitDiff();

        void finishUpdate();
    private:
        int mode;
        const std::vector<Blob<Dtype>*>& learnable_params; 
        std::map<int, std::vector<Dtype*>*> sock2recvCache;
        //std::map<int, int> no2sock;
        //std::map<int, std::thread*> no2handler;
        std::map<int, std::thread*> sock2handler;
        std::vector<Dtype*> diffPtrs;
        std::vector<Dtype*> dataPtrs;
        std::vector<int> dataSize;
        std::vector<int> socks;
        std::thread *listenThread;
        int localSock;
        int currentRecv;
        int currentSend;
        int clientNum;
        bool lastFinish;
        //bool sendFlag = false;
        std::condition_variable finishCond;
        mutable std::mutex countLock;
        //int dataVersion = 0;
        //int lagGap = 1;
        //std::unordered_map<int, int> versionMap;
        //std::unordered_map<int, std::vector<Dtype*>> versionCache;
        
        void initConn();
        void connToServer();
        void closeConn(int sock);
        void handleRequest(int sock);
        void listen();
        void makeListener();
        void mergeDiff();
        void initParams();
};

#endif
