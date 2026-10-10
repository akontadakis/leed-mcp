# Security

leed-mcp runs on your computer and is driven by an AI model. Treat any text the model reads (project files,
scene files, logs) as untrusted. The server is built so that a misled model still cannot:

- read or write outside the folders in `LEED_MCP_ROOTS`;
- follow a link out of those folders;
- pass a Radiance option that names a file or a command;
- start any program other than the pipeline's own Radiance, `epw2wea` and Blender calls.

## Report a problem

Please do not open a public issue for a security problem. Use GitHub's private vulnerability reporting
(the **Security** tab of this repository, then **Report a vulnerability**). Include the steps to reproduce
and the version (`leed-daylight --version`).
