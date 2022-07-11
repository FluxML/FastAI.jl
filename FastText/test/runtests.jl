using FastAI, FastText, ReTest

FastText.runtests([ReTest.fail, ReTest.not(ReTest.pass)])
