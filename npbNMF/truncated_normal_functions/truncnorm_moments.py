import numpy as np
from scipy.special import erfcx

"""
# John P Cunningham
# 2010
#
# Credit given to https://github.com/cunni/epmgp
#
# Calculates the truncated zeroth, first, and second moments of a
# univariate normal distribution.
#
# Much special care is taken to ensure numerical stability over a wide
# range of values, which is particularly important when calculating tail
# probabilities.
#
# there is use of the scaled complementary error function erfcx, etc.
# The actual equations can be found in Jawitz 2004 or Cunningham PhD thesis
# 2009 (Chap 4, Eq 4.6 (note typo in first... alpha and beta are swapped)).
# -This is to be used by Cunningham and Hennig, EP MGP paper.
#
# KEY: normcdf and erf and erfc are unstable when their arguments get big.
# If we are interested in tail probabilities (we are), and if we care about
# logZ more than Z (we do), then normcf/erf/erfc are limited in their
# usefulness.  Instead we can use erfcx(z) = exp(z^2)erfc(z), which has
# some foibles of its own.  The key is to consider that erfcx is stable close to erfc==0, but
# less stable close to erfc==1.  Since the Gaussian is symmetric
# about 0, we can choose to flip the calculation around the origin
# to improve our stability.  For example, if a is -inf, and b is
# -100, then a naive application will return 0.  However, we can
# just flip this around to be erfc(b)= erfc(100), which we can then
# STABLY calculate with erfcx(100)...neat.  This leads to many
# different cases, when either argument is inf or not.
# Also there may appear to be some redundancy here, but it is also
# worth noting that a less detailed application of erfcx can be
# troublesome, as erfcx(-big numbers) = Inf, which wreaks havoc on
# a lot of the calculations.  The cases below treat this when
# necessary.
#
# NOTE: works with singletons or vectors only!
#
# The code is very fast as stands, so loops and other cases are used
# (instead of vectorizing) so that things are most readable.
"""

def moments(lowerB, upperB, mu, sigma):

    """
    lowerB is the lower bound
    upperB is the upper bound
    mu is the mean of the normal distribution (which is truncated)
    sigma is the VARIANCE of the normal distribution (which is truncated)

    """
    """
    # establish bounds
    """
    a = (lowerB - mu)/(np.sqrt(2*sigma))
    b = (upperB - mu)/(np.sqrt(2*sigma))


    """
    # do the stable calculation
    """
    # written out in long format to make clear the steps.  There are steps to
    # take to make this code shorter, but I think it is most readable this way.
    # KEY: The key is to consider that erfcx is stable close to erfc==0, but
    # less stable close to erfc==1.  Since the Gaussian is symmetric
    # about 0, we can choose to flip the calculation around the origin
    # to improve our stability.  For example, if a is -inf, and b is
    # -100, then a naive application will return 0.  However, we can
    # just flip this around to be erfc(b)= erfc(100), which we can then
    # STABLY calculate with erfcx(100)...neat.  This leads to many
    # different cases, when either argument is inf or not.
    # Also there may appear to be some redundancy here, but it is also
    # worth noting that a less detailed application of erfcx can be
    # troublesome, as erfcx(-big numbers) = Inf, which wreaks havoc on
    # a lot of the calculations.  The cases below treat this when
    # necessary.
    # first check for problem cases
    # a problem case
    if np.isinf(a) and np.isinf(b):
        # check the sign
        if np.sign(a) == np.sign(b):
            # then this is integrating from inf to inf, for example.
            #logZhat = -inf
            #meanConst = inf
            #varConst = 0
            logZhat = -np.inf
            Zhat = 0
            muHat = a
            sigmaHat = 0
            entropy = -np.inf
            return logZhat, Zhat, muHat, sigmaHat, entropy
        else:
            #logZhat = 0
            #meanConst = mu
            #varConst = 0
            logZhat = 0
            Zhat = 1
            muHat = mu
            sigmaHat = sigma
            entropy = 0.5*np.log(2*np.pi*np.exp(1)*sigma)
            return logZhat, Zhat, muHat, sigmaHat, entropy
    # a problem case
    elif a > b:
        # these bounds pointing the wrong way, so we return 0 by convention.
        #logZhat = -inf
        #meanConst = 0
        #varConst = 0
        logZhat = -np.inf
        Zhat = 0
        muHat = mu
        sigmaHat = 0
        entropy = -np.inf
        return logZhat, Zhat, muHat, sigmaHat, entropy

    # now real cases follow...
    elif a==-np.inf:
        # then we are integrating everything up to b
        # in infinite precision we just want normcdf(b), but that is not
        # numerically stable.
        # instead we use various erfcx.  erfcx scales very very well for small
        # probabilities (close to 0), but poorly for big probabilities (close
        # to 1).  So some trickery is required.
        if b > 26:
            # then this will be very close to 1... use this goofy expm1 log1p
            # to extend the range up to b==27... 27 std devs away from the
            # mean is really far, so hopefully that should be adequate.  I
            # haven't been able to get it past that, but it should not matter,
            # as it will just equal 1 thereafter.  Slight inaccuracy that
            # should not cause any trouble, but still no division by zero or
            # anything like that.
            # Note that this case is important, because after b=27, logZhat as
            # calculated in the other case will equal inf, not 0 as it should.
            # This case returns 0.
            logZhatOtherTail = np.log(0.5) + np.log(erfcx(b)) - b**2
            logZhat = np.log1p(-np.exp(logZhatOtherTail))

        else:
            # b is less than 26, so should be stable to calculate the moments
            # with a clean application of erfcx, which should work out to
            # an argument almost b==-inf.
            # this is the cleanest case, and the other moments are easy also...
            logZhat = np.log(0.5) + np.log(erfcx(-b)) - b**2

        # the mean/var calculations are insensitive to these calculations, as we do
        # not deal in the log space.  Since we have to exponentiate everything,
        # values will be numerically 0 or 1 at all the tails, so the mean/var will
        # not move.
        # note that the mean and variance are finally calculated below
        # we just calculate the constant here.
        meanConst = -2./erfcx(-b)
        varConst = -2./erfcx(-b)*(upperB + mu)
        #   muHat = mu - (sqrt(sigma/(2*np.pi))*2)./erfcx(-b)
        #   sigmaHat = sigma + mu.^2 - muHat.^2 - (sqrt(sigma/(2*np.pi))*2)./erfcx(-b)*(upperB + mu)

    elif b==np.inf:
        # then we are integrating from a up to Inf, which is just the opposite
        # of the above case.
        if a < -26:
            # then this will be very close to 1... use this goofy expm1 log1p
            # to extend the range up to a==27... 27 std devs away from the
            # mean is really far, so hopefully that should be adequate.  I
            # haven't been able to get it past that, but it should not matter,
            # as it will just equal 1 thereafter.  Slight inaccuracy that
            # should not cause any trouble, but still no division by zero or
            # anything like that.
            # Note that this case is important, because after a=27, logZhat as
            # calculated in the other case will equal inf, not 0 as it should.
            # This case returns 0.
            logZhatOtherTail = np.log(0.5) + np.log(erfcx(-a)) - a**2
            logZhat = np.log1p(-np.exp(logZhatOtherTail))

        else:
            # a is more than -26, so should be stable to calculate the moments
            # with a clean application of erfcx, which should work out to
            # almost inf.
            # this is the cleanest case, and the other moments are easy also...
            logZhat = np.log(0.5) + np.log(erfcx(a)) - a**2

        # the mean/var calculations are insensitive to these calculations, as we do
        # not deal in the log space.  Since we have to exponentiate everything,
        # values will be numerically 0 or 1 at all the tails, so the mean/var will
        # not move.
        meanConst = 2./erfcx(a)
        varConst = 2./erfcx(a)*(lowerB + mu)
        #muHat = mu + (sqrt(sigma/(2*np.pi))*2)./erfcx(a)
        #sigmaHat = sigma + mu.^2 - muHat.^2 + (sqrt(sigma/(2*np.pi))*2)./erfcx(a)*(lowerB + mu)

    else:
        # we have a range from a to b (neither inf), and we need some stable exponent
        # calculations.
        if np.sign(a)==np.sign(b):
            # then we can exploit symmetry in this problem to make the
            # calculations stable for erfcx, that is, with positive arguments:
            # Zerfcx1 = 0.5*(exp(-b.^2)*erfcx(b) - exp(-a.^2)*erfcx(a))
            maxab = max(abs(a),abs(b))
            minab = min(abs(a),abs(b))
            logZhat = np.log(0.5) - minab*2 \
                      + np.log( abs( np.exp(-(maxab**2-minab**2))*erfcx(maxab)\
                      - erfcx(minab)) )

            # now the mean and variance calculations
            # note here the use of the abs and signum functions for flipping the sign
            # of the arguments appropriately.  This uses the relationship
            # erfc(a) = 2 - erfc(-a).
            meanConst = 2*np.sign(a)*(1/((erfcx(abs(a)) \
                                      - np.exp(a**2-b**2)*erfcx(abs(b))))\
                                      - 1/((np.exp(b**2-a**2)*erfcx(abs(a))\
                                      - erfcx(abs(b)))))
            varConst =  2*np.sign(a)*((lowerB+mu)/((erfcx(abs(a))\
                        - np.exp(a**2-b**2)*erfcx(abs(b))))\
                        - (upperB+mu)/((np.exp(b**2-a**2)*erfcx(abs(a))\
                        - erfcx(abs(b)))))

        else:
            # then the signs are different, which means b>a (upper>lower by definition), and b>=0, a<=0.
            # but we want to take the bigger one (larger magnitude) and make it positive, as that
            # is the numerically stable end of this tail.
            if abs(b) >= abs(a):
                if a >= -26:

                    # do things normally
                    logZhat = np.log(0.5) - a**2 + np.log( erfcx(a)\
                              - np.exp(-(b**2 - a**2))*erfcx(b) )

                    # now the mean and var
                    meanConst = 2*(1/((erfcx(a)\
                                - np.exp(a**2-b**2)*erfcx(b)))\
                                - 1/((np.exp(b**2-a**2)*erfcx(a) - erfcx(b))))
                    varConst = 2*((lowerB+mu)/((erfcx(a)\
                               - np.exp(a**2-b**2)*erfcx(b)))\
                               - (upperB+mu)/((np.exp(b**2-a**2)*erfcx(a)\
                               - erfcx(b))))

                else:
                    # a is too small and the calculation will be unstable, so
                    # we just put in something very close to 2 instead.
                    # Again this uses the relationship
                    # erfc(a) = 2 - erfc(-a). Since a<0 and b>0, this
                    # case makes sense.  This just says 2 - the right
                    # tail - the left tail.
                    logZhat = np.log(0.5) + np.log( 2 - np.exp(-b**2)*erfcx(b)\
                              - np.exp(-a**2)*erfcx(-a) )

                    # now the mean and var
                    meanConst = 2*(1/((erfcx(a) - np.exp(a**2-b**2)*erfcx(b)))\
                                - 1/(np.exp(b**2)*2 - erfcx(b)))
                    varConst = 2*((lowerB+mu)/((erfcx(a)\
                               - np.exp(a**2-b**2)*erfcx(b)))\
                               - (upperB+mu)/(np.exp(b**2)*2 - erfcx(b)))

            else: # abs(a) is bigger than abs(b), so we reverse the calculation...
                if b <= 26:

                    # do things normally but mirrored across 0
                    logZhat = np.log(0.5) - b**2 + np.log( erfcx(-b)\
                              - np.exp(-(a**2 - b**2))*erfcx(-a))

                    # now the mean and var
                    meanConst = -2*(1/((erfcx(-a)\
                                - np.exp(a**2-b**2)*erfcx(-b)))\
                                - 1/((np.exp(b**2-a**2)*erfcx(-a)\
                                - erfcx(-b))))
                    varConst = -2*((lowerB+mu)/((erfcx(-a)\
                               - np.exp(a**2-b**2)*erfcx(-b)))\
                               - (upperB+mu)/((np.exp(b**2-a**2)*erfcx(-a)\
                               - erfcx(-b))))

                else:

                    # b is too big and the calculation will be unstable, so
                    # we just put in something very close to 2 instead.
                    # Again this uses the relationship
                    # erfc(a) = 2 - erfc(-a). Since a<0 and b>0, this
                    # case makes sense. This just says 2 - the right
                    # tail - the left tail.
                    logZhat = np.log(0.5)\
                              + np.log( 2 - np.exp(-a**2)*erfcx(-a)\
                              - np.exp(-b**2)*erfcx(b) )

                    # now the mean and var
                    meanConst = -2*(1/(erfcx(-a) - np.exp(a**2)*2)\
                                - 1/(np.exp(b**2-a**2)*erfcx(-a) - erfcx(-b)))
                    varConst = -2*((lowerB + mu)/(erfcx(-a)\
                               - np.exp(a**2)*2)\
                               - (upperB + mu)/(np.exp(b**2-a**2)*erfcx(-a)\
                               - erfcx(-b)))

            # the above four cases (diff signs x stable/unstable) can be
            # collapsed into two cases by tracking the sign of the maxab
            # and sign of the minab (the min and max of abs(a) and
            # abs(b)), but that is a bit less clear, so we
            # leave it fleshed out above.



    """
    # finally, calculate the returned values
    """
    # logZhat is already calculated, as are meanConst and varConst.
    # no numerical precision in Zhat
    Zhat = np.exp(logZhat)
    # make the mean
    muHat = mu + meanConst*np.sqrt(sigma/(2*np.pi))
    # make the var
    sigmaHat = sigma + varConst*np.sqrt(sigma/(2*np.pi)) + mu**2 - muHat**2
    # make entropy
    entropy = 0.5*((meanConst*np.sqrt(sigma/(2*np.pi)))**2
                + sigmaHat - sigma)/sigma\
              + logZhat + np.log(np.sqrt(2*np.pi*np.exp(1)))\
              + np.log(np.sqrt(sigma))
    return logZhat, Zhat, muHat, sigmaHat, entropy
