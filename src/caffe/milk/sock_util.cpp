#include <iostream>
#include <cstring>

#include <fcntl.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <unistd.h> 
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <arpa/inet.h>

#include "caffe/milk/sock_util.hpp"

void error(const char *msg) {
    perror(msg);
    exit(1);
}

int makeServerConn(int port) {
    struct sockaddr_in serv_addr;
    int sockfd = socket(AF_INET, SOCK_STREAM, 0); 
    if (sockfd < 0) error("ERROR opening socket");
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(port);
    int bOptval=1;  
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char *)&bOptval, sizeof(bOptval));
    setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (void *)&bOptval, sizeof(bOptval));
    if(bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) error("ERROR on binding");
    listen(sockfd, 5); 
    return sockfd;
}


int acceptAConn(int sock) {
    struct sockaddr_in cli_addr;
    socklen_t clilen = sizeof(cli_addr);

    char remoteAddr[16];
    int clientSock = accept(sock,  (struct sockaddr *) &cli_addr, &clilen);
    strcpy(remoteAddr, inet_ntoa(cli_addr.sin_addr));
    std::cout << "Get a conn, ip: " << remoteAddr << std::endl;
    return clientSock;
}

int makeConn(const std::string &hostname, int port) {
    struct hostent* server = gethostbyname(hostname.c_str());
    if(!server) {
        printf("Hostname invalide!\n");
        return -1;
    }
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in serv_addr;
    bzero((char*) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char*) server->h_addr, (char*) &serv_addr.sin_addr.s_addr, server->h_length);
    serv_addr.sin_port = htons(port);

    if(connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        printf("Connect error\n");
    }
    int i = 1;
    setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (void *)&i, sizeof(i));
    //fcntl(sockfd, F_SETFL, O_NONBLOCK);
    return sockfd;
}

int sendAll(int sockfd, const void *data, int allSize) {
    int sum = 0;
    while(sum < allSize) {
        int t = send(sockfd, data + sum, allSize - sum, 0);
        if(t > 0) { sum += t; }
        else if(t <= 0 && errno != EAGAIN) return t;
        //if(t <= 0) return t;
        //sum += t;
    }
    return sum;
}

int recvAll(int sockfd, void *data, int allSize) {
    int sum = 0;
    while(sum < allSize) {
        int t = recv(sockfd, data + sum, allSize - sum, 0);
        if(t > 0) { sum += t; }
        else if(t <= 0 && errno != EAGAIN) return t;
        //if(t <= 0) return t;
        //sum += t;
    }
    return sum;
}  
