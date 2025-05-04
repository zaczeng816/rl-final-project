/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Trust the proxy headers
  async headers() {
    return [];
  },
  // Trust the proxy
  async rewrites() {
    return [];
  },
};

module.exports = nextConfig; 