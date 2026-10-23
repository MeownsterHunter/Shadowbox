# Questions and answers

Plain answers about what this project is and how you may use it. [LICENSE.md](LICENSE.md) has a table that says which license fits which use. The full license texts are in the LICENSES folder, and they are what counts if this page ever reads differently.

## What is this for?

Research and curiosity. It is an attempt to understand how a console's graphics interface behaves and to rebuild it on top of Vulkan, in the open, so that others can learn from it and build on it.

## Can I use it?

Yes. There are three ways, and you choose one:

1. **Noncommercial use is free.** Personal projects, hobby work, study, teaching, research, and use by charities, schools, public bodies and similar organisations.
2. **GPL and AGPL projects can include it for free.** If your project is under version 3 of the GPL or AGPL (or GPL version 2 "or later"), Shadowbox can be part of it under the AGPL.
3. **Closed commercial use is paid.** If you want to make money with it and keep your own source closed, you need a commercial license.

## Can I change it?

Yes. Under the noncommercial license you can modify it however you like for noncommercial purposes and share your changes on the same terms. Under the AGPL you can modify it and share your changes under the AGPL. Either way, pass the license terms and the copyright notice along with it so the next person knows the terms too. Each license says exactly how.

## Why is it not simply free for everything?

The work is shared so people can learn from it and enjoy it. It is not shared so that someone can take it for nothing, put it in a closed product and sell that to people who have no idea what is inside or who made it. Low-effort products built on other people's unpaid work are bad for the people who buy them and bad for the people who did the work.

So closed commercial use has a price. Anyone who wants to make money with Shadowbox without sharing their own source can do so by getting a commercial license. See [COMMERCIAL.md](COMMERCIAL.md).

## What counts as commercial?

Roughly, anything done to make money or as part of a business. Selling a device or program that includes Shadowbox, or bundling it with something you sell, are the clear cases. A hobby project that takes no money is not commercial.

The noncommercial license also names kinds of organisation that may use it whatever their funding: charities, schools and universities, public research bodies, public safety and health organisations, environmental groups and government institutions. Its text has the exact list.

If you are in between, ask before you ship.

## Why are there two free licenses? Is one not enough?

Each covers people the other leaves out. The noncommercial license is the simple one: its main condition is "not for commercial use". But projects under the GPL are not allowed to take in code that carries a restriction like that. So the AGPL is offered as well. It is the GPL's sibling, GPL version 3 projects are allowed to combine with it, and it lets those projects include Shadowbox too.

## Why the AGPL and not the plain GPL?

The plain GPL only asks for source when the software itself is handed to someone. A company could run it on its own servers, charge for the service, and share nothing. The AGPL closes that: people who use the software over a network must be offered its source too.

## Can a company use the AGPL option and make money without paying?

Only by following the AGPL in full. Whatever they hand to customers, or let people use over a network, that combines Shadowbox with their own code has to come with its source under the same kind of license. For software shipped in consumer products, the license in most cases also requires what a buyer needs to install a changed version. A company that does all that is sharing its work back, which is fair. A company that wants to keep things closed needs the commercial license.

## I run a free, community emulator or tool. Can I use it?

Yes. If your project is noncommercial, use the noncommercial license. If your project is under the GPL or AGPL, use the AGPL. Either way you may bundle Shadowbox. The one case that fits neither is a project under GPL version 2 only. If that is you, open an issue.

## Is this open source?

Under the AGPL option, yes: the AGPL is an open source and free software license. The noncommercial option is not open source by the usual definition, because that definition does not allow a noncommercial restriction. You only need one of the two, so pick the one that suits you.

## Can I contribute?

Yes, and thank you. Because commercial licenses are offered, contributors are asked to agree to a short contributor agreement so the project is allowed to include their work under the two public licenses and under commercial licenses. See [CONTRIBUTING.md](CONTRIBUTING.md) and [CLA.md](CLA.md). The clean-room rules in [CLEAN-ROOM.md](CLEAN-ROOM.md) apply to every contribution.

## Does it include anything from a console maker or from games?

No. No code, headers, documentation, shaders, keys, firmware or game data from anyone else. It was written from scratch under clean-room rules, and it does not decrypt or unlock anything. See [LEGAL.md](LEGAL.md).

## Will it run my games?

Not by itself, and not yet. It is one piece that an emulator or similar tool would use, and at the moment it does not render anything. The README says where things stand.
