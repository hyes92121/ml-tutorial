# Tutorial for how to use git in the ML class



#### 1. How to get your own repository

After creating our own repository, we would want to have a local copy of it, and wish that any changes made to it would be also made to the remote one. 

So naturally, we need to "clone" the remote repository to local. To do that, type

`git clone https://github.com/YOUR_USERNAME/ML2019SPRING`

in your command line. 

A new folder named `ML2019SPRING` will be created in your working directory. This is your local in-class repository and should be synced with the remote one at all times. 

#### 2. How to add files to commit

To perform a commit, one first needs to "add" the desired files to the local index.

Type `git add [LIST OF FILES]`, or `git add -A` to add all files to the index.

#### 3. How to commit a change

Any changes made to the local repository needs to be committed before being "pushed", or "updated" to the remote one.

To view changes being made to the repository, type `git status`.

If the updated files are correct, we should then proceed to commit these changes.

Type `git commit -m "YOUR COMMIT MESSAGE"` to commit the update. 

#### 4. How to push to remote

Now that we've committed our changes, it's time for us to push the updated files to the remote repository.

Type `git push` to push your files to remote. 



That's it. You've learned how to update and push to your own repository.

