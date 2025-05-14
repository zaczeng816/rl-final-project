/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Trust the proxy headers
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          {
            key: "X-Robots-Tag",
            value: "index, follow",
          },
          {
            key: "X-Content-Type-Options",
            value: "nosniff",
          },
        ],
      },
    ];
  },
  // Trust the proxy
  async rewrites() {
    return [];
  },
};

module.exports = nextConfig;
