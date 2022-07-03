using FastAI, FastMakie, ReTest

FastMakie.runtests([ReTest.fail, ReTest.not(ReTest.pass)])
