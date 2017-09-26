#include <chrono>
#include <cstdio>

#include "caffe/milk/sync_manager.hpp"
#include "caffe/milk/sock_util.hpp"

template<typename Dtype>
SyncManager<Dtype>::SyncManager(int mode_, int num, const std::vector<Blob<Dtype>*>& learnable_params_): 
    mode(mode_), 
    learnable_params(learnable_params_),
    currentRecv(0),
    currentSend(0),
    clientNum(num),
    lastFinish(true)
{
    mode = mode;
    initParams();
    initConn();
}

template<typename Dtype>
SyncManager<Dtype>::~SyncManager() {
    if(mode == 1) closeConn(localSock);
}

template<typename Dtype>
void SyncManager<Dtype>::initParams() {
    switch(Caffe::mode()) {
        case Caffe::CPU:
            for(auto l : learnable_params) {
                dataPtrs.push_back(l->mutable_cpu_data());
                diffPtrs.push_back(l->mutable_cpu_diff());
                dataSize.push_back(l->count() * sizeof(Dtype));
            }
            break;
        case Caffe::GPU:
            for(auto l : learnable_params) {
                dataPtrs.push_back(l->mutable_gpu_data());
                diffPtrs.push_back(l->mutable_gpu_diff());
                dataSize.push_back(l->count() * sizeof(Dtype));
                printf("layer size is %d\n", l->count() * sizeof(Dtype));
            }
            break;
    }
}


template<typename Dtype>
void SyncManager<Dtype>::initConn() {
    if(mode == SERVER_MODE) {
        localSock = makeServerConn(10088);
        printf("server started\n");
        makeListener();
        printf("listen started\n");
    } else {
        connToServer();
    }
}

template<typename Dtype>
void SyncManager<Dtype>::connToServer() {
    localSock = makeConn("127.0.0.1", 10088);
}


template<typename Dtype>
void SyncManager<Dtype>::makeListener() {
    listenThread = new std::thread(&SyncManager<Dtype>::listen, this);
    listenThread->detach();
}

template<typename Dtype>
void SyncManager<Dtype>::listen() {
    int sock;
    //int NO;
    while(1) {
        sock = acceptAConn(localSock);
        //recvAll(sock, &NO, sizeof(NO));
        socks.push_back(sock);
        sock2handler[sock] = new std::thread(&SyncManager<Dtype>::handleRequest, this, sock);
    }
}

// for server
template<typename Dtype>
void SyncManager<Dtype>::closeConn(int sock) {
    int flag = CLOSE_CONN;
    sendAll(sock, &flag, sizeof(flag));
    //recvAll(sock, &flag, sizeof(flag));
    close(sock);
}

template<typename Dtype>
void SyncManager<Dtype>::handleRequest(int sock) {
    int signal;
    while(1) {
        recvAll(sock, &signal, sizeof(signal));
        //printf("get a signal %d\n", signal);
        switch(signal) {
            case CLOSE_CONN:
                closeConn(sock);
                return;
            case PULL_FULL_MODEL:
                sendModel(sock);
                break;
            case PUSH_FULL_DIFF:
                getDiff(sock);
                break;
            default:
                break;
        }
    }
    //printf("handler exit\n");
}

template<typename Dtype>
void SyncManager<Dtype>::pushDiff() {
    //printf("send model diff to server\n");
    int flag = PUSH_FULL_DIFF;
    sendAll(localSock, &flag, sizeof(flag));
    for(size_t i = 0; i < learnable_params.size(); ++i) {
        //printf("send learnable layer diff id %d size %d\n", i, size);
        sendAll(localSock, diffPtrs[i], dataSize[i]);
    }
}

template<typename Dtype>
void SyncManager<Dtype>::pullModel() {
    //printf("Get model from server\n");
    int flag = PULL_FULL_MODEL;
    sendAll(localSock, &flag, sizeof(flag));
    for(size_t i = 0; i < learnable_params.size(); ++i) {
        //printf("recv learnable layer id %d size %d\n", i, dataSize[i]);
        recvAll(localSock, dataPtrs[i], dataSize[i]);
    }
    //printf("Get model from server done.\n");
}

template<typename Dtype>
void SyncManager<Dtype>::getDiff(int sock) {
    //printf("Get diff from client\n");
    std::vector<Dtype*>* tmp = new std::vector<Dtype*>();
    for(size_t i = 0; i < learnable_params.size(); ++i) {
        int size = dataSize[i];
        Dtype* buff = (Dtype*) malloc(size);
        recvAll(sock, buff, size);
        //Dtype* mdata = learnable_params[i]->mutable_cpu_diff();
        //for(int j = 0; j < learnable_params[i]->count(); ++j) {
        //    mdata[j] += buff[j] / clientNum;
        //}
        tmp->push_back(buff);
        //free(buff);
    }
    countLock.lock();
    sock2recvCache[sock] = tmp;
    ++currentRecv;
    countLock.unlock();
    finishCond.notify_all();
}

template<typename Dtype>
void SyncManager<Dtype>::sendModel(int sock) {
    //printf("send model to client\n");
    std::unique_lock<std::mutex> lck(countLock);
    finishCond.wait(lck, [this]{return lastFinish;});
    for(size_t i = 0; i < learnable_params.size(); ++i) {
        //printf("send learnable layer id %d size %d\n", i, dataSize[i]);
        sendAll(sock, dataPtrs[i], dataSize[i]);
    }
    ++currentSend;
    if(currentSend == clientNum) {
        lastFinish = false;
        currentSend = 0;
    }
    //printf("send model to client done\n");
}

template<typename Dtype>
void SyncManager<Dtype>::waitDiff() {
    std::unique_lock<std::mutex> lck(countLock);
    finishCond.wait(lck, [this]{return currentRecv == clientNum;});
    mergeDiff();
}

template<typename Dtype>
void SyncManager<Dtype>::finishUpdate() {
    currentRecv = 0;
    //for(size_t i = 0; i < learnable_params.size(); ++i) {
    //    auto blob = learnable_params[i];
    //    caffe::caffe_set(blob->count(), static_cast<Dtype>(0), diffPtrs[i]);
    //}
    lastFinish = true;
    finishCond.notify_all();
}

template<typename Dtype>
void SyncManager<Dtype>::mergeDiff() {
    // set diff to zero.
    //for(size_t i = 0; i < learnable_params.size(); ++i) {
    //    auto blob = learnable_params[i];
    //    caffe::caffe_set(blob->count(), static_cast<Dtype>(0), blob->mutable_cpu_diff());
    //}
    for(auto it = sock2recvCache.begin(); it !=  sock2recvCache.end(); ++it) {
        for(size_t i = 0; i < learnable_params.size(); ++i) {
            Dtype* mdata = diffPtrs[i];
            for(int j = 0; j < learnable_params[i]->count(); ++j) {
                mdata[j] += (*it->second)[i][j] / clientNum;
            }
        }
        free(it->second);
    }
    sock2recvCache.clear();
}

template class SyncManager<float>;
template class SyncManager<double>;
template class SyncManager<int>;
