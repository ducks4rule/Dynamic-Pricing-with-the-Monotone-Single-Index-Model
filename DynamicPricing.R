theta0 = c(1,2)

n = 10000
feat = NULL
for(j in 1:10000){
feat = rbind(feat, rnorm(n=2, mean=1, sd=1))

}

noise = runif(n=10000, min=-0.5, max =0.5)

vt = feat%*%theta0 + noise

pt =  vt +  rnorm(10000, sd=0.5)

yt = ifelse(pt <= vt, 1, 0)

library(Iso)

#Suppose we know the truth theta0

Listy = list(coor = pt - feat%*%theta0, ind = yt)

Fest = pava(y =  1-Listy$ind[order(Listy$coor)], decreasing=FALSE)

plot(sort(Listy$coor), Fest, xlab="", ylab="", type="p")


crit = NULL
Theta = NULL

for(i in (-10:10)){
   for(j in (-10:10)){

        theta.ij = c(i, j)
        Theta = rbind(Theta, theta.ij)
        List.ij = list(coor = pt - feat%*%theta.ij, ind = yt)
        Fest.ij = pava(y =  1-List.ij$ind[order(List.ij$coor)], decreasing=FALSE)
        crit.ij = sum((Fest.ij - (1-List.ij$ind[order(List.ij$coor)]))^2)
        crit = c(crit, crit.ij)
    }
}

plot(1:441, crit, xlab="", ylab="", type="b")
which.min(crit) = 244
Theta[244,]  = c(1,2) ...!

################



