#include <chrono>
#include <cstdio>

#include "caffe/milk/sync_manager.hpp"
#include "caffe/milk/sock_util.hpp"

template<typename Dtype>
SyncManager<Dtype>::SyncManager(int mode_, int num, const std::vector<Blob<Dtype>*>& learnable_params_): 
    mode(mode_), 
    learnable_params(learnable_params_),
    diffRecv(0),
    clientNum(num),
    lastUpdateFinish(true)
{
    mode = mode;
    printf("have %d clients\n", clientNum);
    for(size_t i = 0; i < learnable_params.size(); ++i) {
        dataSize.push_back(learnable_params[i]->count() * sizeof(Dtype));
    }
    initConn();
}

template<typename Dtype>
SyncManager<Dtype>::~SyncManager() {
    if(mode == 1) closeConn(localSock);
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
        sendAll(localSock, learnable_params[i]->cpu_diff(), dataSize[i]);
    }
    //printf("send model diff to server done\n");
}

template<typename Dtype>
void SyncManager<Dtype>::pullModel() {
    //printf("Get model from server\n");
    int flag = PULL_FULL_MODEL;
    sendAll(localSock, &flag, sizeof(flag));
    for(size_t i = 0; i < learnable_params.size(); ++i) {
        //printf("recv learnable layer diff id %d size %d\n", i, learnable_params[i]->count() * sizeof(Dtype));
        recvAll(localSock, learnable_params[i]->mutable_cpu_data(), dataSize[i]);
    }
    //printf("Get model from server done.\n");
}

template<typename Dtype>
void SyncManager<Dtype>::getDiff(int sock) {
    //printf("Get diff from client\n");
    lastUpdateFinish = false;
    std::vector<Dtype*>* tmp = new std::vector<Dtype*>();
    for(size_t i = 0; i < learnable_params.size(); ++i) {
        int size = dataSize[i];
        Dtype* buff = (Dtype*) malloc(size);
        recvAll(sock, buff, size);
        tmp->push_back(buff);
    }

    diffRecvLock.lock();
    sock2recvCache[sock] = tmp;
    ++diffRecv;
    diffRecvLock.unlock();

    diffRecvFinishCond.notify_all();
    //printf("Get diff from client done. currentRecv, %d\n", currentRecv);
}

// Only after update finish, the model can be send.
template<typename Dtype>
void SyncManager<Dtype>::sendModel(int sock) {
    //printf("send model to client\n");
    std::unique_lock<std::mutex> lck(lastUpdateFinishLock);
    lastUpdateFinishCond.wait(lck, [this]{ return lastUpdateFinish; });
    for(size_t i = 0; i < learnable_params.size(); ++i) {
        //printf("send learnable layer id %d size %d\n", i, learnable_params[i]->count() * sizeof(Dtype));
        sendAll(sock, learnable_params[i]->cpu_data(), dataSize[i]);
    }
    //printf("send model to client done\n");
}

template<typename Dtype>
void SyncManager<Dtype>::waitDiff() {
    std::unique_lock<std::mutex> lck(lastUpdateFinishLock);
    diffRecvFinishCond.wait(lck, [this]{ return diffRecv == clientNum;});
    mergeDiff();
}

template<typename Dtype>
void SyncManager<Dtype>::finishUpdate() {
    diffRecv = 0;
    lastUpdateFinish = true;
    lastUpdateFinishCond.notify_all();
}

template<typename Dtype>
void SyncManager<Dtype>::mergeDiff() {
    //printf("%d client num\n", clientNum);
    for(auto it = sock2recvCache.begin(); it !=  sock2recvCache.end(); ++it) {
        for(size_t i = 0; i < learnable_params.size(); ++i) {
            Dtype* mdata = learnable_params[i]->mutable_cpu_diff();
            for(int j = 0; j < learnable_params[i]->count(); ++j) {
                mdata[j] += (*it->second)[i][j] / clientNum;
                //mdata[j] += (*it->second)[i][j] / clientNum;
            }
        }
        free(it->second);
    }
    sock2recvCache.clear();
}

template class SyncManager<float>;
template class SyncManager<double>;
template class SyncManager<int>;
