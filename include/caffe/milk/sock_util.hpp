#ifndef __SOCK_UTIL_H__
#define __SOCK_UTIL_H__

#include <string>

void error(const char *msg);

int makeServerConn(int port);

int acceptAConn(int sock);

int makeConn(const std::string &hostname, int port);

int sendAll(int sockfd, const void *data, int allSize);

int recvAll(int sockfd, void *data, int allSize);

#endif
