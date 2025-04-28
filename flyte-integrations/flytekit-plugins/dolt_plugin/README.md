# Dolt

The `DoltTable` plugin is a wrapper that uses [Dolt](https://github.com/dolthub/dolt) to move data between
`pandas.DataFrame`'s at execution time and database tables at rest.

## Installation

The dolt plugin and dolt command line tool are required to run these examples:

```shell
$ pip install flytekitplugins.dolt
$ sudo bash -c 'curl -L https://github.com/dolthub/dolt/releases/latest/download/install.sh | sudo bash'
```

Dolt requires a user configuration to run `init`:

```
$ dolt config --global --add user.email <email>
$ dolt config --global --add user.name <name>
```

These demos assume a `foo` database has been created locally:

```
$ mkdir foo
$ cd foo
$ dolt init
```
